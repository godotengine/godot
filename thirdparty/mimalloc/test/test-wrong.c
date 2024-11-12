/* ----------------------------------------------------------------------------
Copyright (c) 2018-2020, Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* test file for valgrind/asan support.

   VALGRIND:
   ----------
   Compile in an "out/debug" folder:

   > cd out/debug
   > cmake ../.. -DMI_TRACK_VALGRIND=1
   > make -j8

   and then compile this file as:

   > gcc -g -o test-wrong -I../../include ../../test/test-wrong.c libmimalloc-valgrind-debug.a -lpthread

   and test as:

   > valgrind ./test-wrong

   
   ASAN
   ----------
   Compile in an "out/debug" folder:

   > cd out/debug
   > cmake ../.. -DMI_TRACK_ASAN=1
   > make -j8

   and then compile this file as:

   > clang -g -o test-wrong -I../../include ../../test/test-wrong.c libmimalloc-asan-debug.a -lpthread -fsanitize=address -fsanitize-recover=address

   and test as:

   > ASAN_OPTIONS=verbosity=1:halt_on_error=0 ./test-wrong


*/
#include <stdio.h>
#include <stdlib.h>
#include "mimalloc.h"

#ifdef USE_STD_MALLOC
# define mi(x) x
#else
# define mi(x) mi_##x
#endif

int main(int argc, char** argv) {
  int* p = (int*)mi(malloc)(3*sizeof(int));

  int* r = (int*)mi_malloc_aligned(8,16);
  mi_free(r);

  // illegal byte wise read
  char* c = (char*)mi(malloc)(3);
  printf("invalid byte: over: %d, under: %d\n", c[4], c[-1]);
  mi(free)(c);

  // undefined access
  int* q = (int*)mi(malloc)(sizeof(int));
  printf("undefined: %d\n", *q);

  // illegal int read
  printf("invalid: over: %d, under: %d\n", q[1], q[-1]);

  *q = 42;

  // buffer overflow
  q[1] = 43;

  // buffer underflow
  q[-1] = 44;

  mi(free)(q);

  // double free
  mi(free)(q);

  // use after free
  printf("use-after-free: %d\n", *q);

  // leak p
  // mi_free(p)
  return 0;
}