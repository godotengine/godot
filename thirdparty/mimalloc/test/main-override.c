#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <mimalloc-override.h>

int main() {
  mi_version();       // ensure mimalloc library is linked
  void* p1 = malloc(78);
  void* p2 = malloc(24);
  free(p1);
  p1 = malloc(8);
  //char* s = strdup("hello\n");
  free(p2);
  p2 = malloc(16);
  p1 = realloc(p1, 32);
  free(p1);
  free(p2);
  //free(s);
  //mi_collect(true);

  /* now test if override worked by allocating/freeing across the api's*/
  //p1 = mi_malloc(32);
  //free(p1);
  //p2 = malloc(32);
  //mi_free(p2);
  p1 = malloc(24);
  p2 = reallocarray(p1, 16, 16);
  free(p2);
  p1 = malloc(24);
  assert(reallocarr(&p1, 16, 16) == 0);
  free(p1);
  mi_stats_print(NULL);
  return 0;
}
