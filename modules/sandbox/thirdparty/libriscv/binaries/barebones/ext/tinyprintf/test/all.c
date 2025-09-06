#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "tinyprintf.h"

char expected_buffer[1000];
char actual_buffer[1000];

int passes = 0, failures = 0;

void check(const char *expr, int expected_sz, int actual_sz)
{
  int size_pass = expected_sz == actual_sz;
  int content_pass = strcmp(expected_buffer, actual_buffer) == 0;
  int pass = size_pass && content_pass;

  if (pass) {
    passes++;
    printf("PASS       printf(%s)\n", expr);
    printf("Output     %s", actual_buffer);
  } else {
    failures++;
    printf("FAIL       printf(%s)\n", expr);

    if (!content_pass) {
      printf("libc       %s\n", expected_buffer);
      printf("tinyprintf %s\n", actual_buffer);
    }
  }
  printf("\n");
}

#define TPRINTF(expr...) do { \
    int expected_sz = snprintf(expected_buffer, sizeof(expected_buffer), expr);\
    int actual_sz = tfp_snprintf(actual_buffer, sizeof(actual_buffer), expr);\
    check(#expr, expected_sz, actual_sz);\
} while(0)

void main()
{
  TPRINTF("d1=%016llx d2=%016lx d3=%02x d4=%02X 42=%03d",
          (long long unsigned) 0xd1, (long unsigned) 0xd2, 0xd3, 0xd4, 42);
  TPRINTF("d1=%04x d2=%06x d3=%08x %%100", 0xd1, 0xd2, 0xd3);
  TPRINTF("|%-14s| |%-16s| d2=%2x |%-30s|", "str14", "str16", 0xd2,
          "12345678901234567890123456789012345");
  TPRINTF("|%4s|", "string4");
  TPRINTF("|%-4s|", "string4");
  TPRINTF("42=%3d d1=%4.4x |%4s| d2=%8.8x", 42, 0xd1, "string4", 0xd2);
  TPRINTF("42=%3d d1=%4.4x |%-4s| d2=%8.8x", 42, 0xd1, "string4", 0xd2);
  TPRINTF("84=%d 21=%ds |%s| |%sOK| d1=%x d2=%#x",
          84, 21, "hello", "fine", 0xd1, 0xd2);
  TPRINTF("42=% 3d d1=%4x |%10s| d2=%3.3x", 42, 0xd1, "string4", 0xd2);

  TPRINTF("%lld", LLONG_MIN);
  TPRINTF("%lld", LLONG_MAX);
  TPRINTF("%llu", ULLONG_MAX);
  TPRINTF("%llx", LLONG_MIN);
  TPRINTF("%llx", LLONG_MAX);
  TPRINTF("%llx", ULLONG_MAX);

  TPRINTF("d1=%.1x", 0xd1);
  TPRINTF("d1=%4.1x", 0xd1);
  TPRINTF("d1=%4.x", 0xd1);
  TPRINTF("d1=%4.4x", 0xd1);
  TPRINTF("d1=%04x", 0xd1);

  {
    char blah[256];
    TPRINTF("a=%zd", sizeof(blah));
    TPRINTF("a=%zu", sizeof(blah));
    TPRINTF("a=%zi", sizeof(blah));
    TPRINTF("a=0x%zx", sizeof(blah));
  }

  {
    int in_stack;
    TPRINTF("Adddress of main: %p", main);
    TPRINTF("Adddress of stack variable: %p", &in_stack);
  }

  {
    char buf[] = "0123456789";
    TPRINTF("%*s", 5, &buf[5]);         /* minimum length, too long string */
    TPRINTF("%.*s", 5, buf);            /* maximum length, too long string */
    TPRINTF("%.*s", 5, &buf[8]);        /* maximum length, too short string */
    TPRINTF("%*.*s", 5, 5, &buf[8]);    /* minimum and maximum, too short string*/
    TPRINTF("%*.*s", 5, 1, &buf[8]);
    TPRINTF("%*.*s", 5, 5, buf);

    TPRINTF("%-*d", 5, 123);
    TPRINTF("%*d", 5, 123);
  }

  printf("Sizeof\n");
  printf("  char      %zd\n", sizeof(char));
  printf("  int       %zd\n", sizeof(int));
  printf("  long      %zd\n", sizeof(long));
  printf("  long long %zd\n", sizeof(long long));
  printf("  void*     %zd\n", sizeof(void *));
  printf("\n");

  printf("Summary\n");
  printf("  passes:   %d\n", passes);
  printf("  failures: %d\n", failures);
}
