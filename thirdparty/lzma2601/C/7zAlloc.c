/* 7zAlloc.c -- Allocation functions for 7z processing
2023-03-04 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include <stdlib.h>

#include "7zAlloc.h"

/* #define SZ_ALLOC_DEBUG */
/* use SZ_ALLOC_DEBUG to debug alloc/free operations */

#ifdef SZ_ALLOC_DEBUG

/*
#ifdef _WIN32
#include "7zWindows.h"
#endif
*/

#include <stdio.h>
static int g_allocCount = 0;
static int g_allocCountTemp = 0;

static void Print_Alloc(const char *s, size_t size, int *counter)
{
  const unsigned size2 = (unsigned)size;
  fprintf(stderr, "\n%s count = %10d : %10u bytes; ", s, *counter, size2);
  (*counter)++;
}
static void Print_Free(const char *s, int *counter)
{
  (*counter)--;
  fprintf(stderr, "\n%s count = %10d", s, *counter);
}
#endif

void *SzAlloc(ISzAllocPtr p, size_t size)
{
  UNUSED_VAR(p)
  if (size == 0)
    return 0;
  #ifdef SZ_ALLOC_DEBUG
  Print_Alloc("Alloc", size, &g_allocCount);
  #endif
  return malloc(size);
}

void SzFree(ISzAllocPtr p, void *address)
{
  UNUSED_VAR(p)
  #ifdef SZ_ALLOC_DEBUG
  if (address)
    Print_Free("Free ", &g_allocCount);
  #endif
  free(address);
}

void *SzAllocTemp(ISzAllocPtr p, size_t size)
{
  UNUSED_VAR(p)
  if (size == 0)
    return 0;
  #ifdef SZ_ALLOC_DEBUG
  Print_Alloc("Alloc_temp", size, &g_allocCountTemp);
  /*
  #ifdef _WIN32
  return HeapAlloc(GetProcessHeap(), 0, size);
  #endif
  */
  #endif
  return malloc(size);
}

void SzFreeTemp(ISzAllocPtr p, void *address)
{
  UNUSED_VAR(p)
  #ifdef SZ_ALLOC_DEBUG
  if (address)
    Print_Free("Free_temp ", &g_allocCountTemp);
  /*
  #ifdef _WIN32
  HeapFree(GetProcessHeap(), 0, address);
  return;
  #endif
  */
  #endif
  free(address);
}
