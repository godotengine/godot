
#define TINYPRINTF_DEFINE_TFP_PRINTF 1
#define TINYPRINTF_DEFINE_TFP_SPRINTF 0
#define TINYPRINTF_OVERRIDE_LIBC 0

#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "tinyprintf.h"

/* Clear unused warnings for actually unused variables */
#define UNUSED(x) (void)(x)

#define TPRINTF(expr...) \
  ({ printf("libc_printf(%s) -> ", #expr); printf(expr); \
     printf(" tfp_printf(%s) -> ", #expr); tfp_printf(expr); })

static void stdout_putf(void *unused, char c)
{
  UNUSED(unused);
  putchar(c);
}

int main()
{
  init_printf(NULL, stdout_putf);

  printf("Fun with printf and %%!\n");

  TPRINTF("d1=%016llx d2=%016lx d3=%02x d4=%02X 42=%03d\n",
          (long long unsigned)0xd1, (long unsigned)0xd2, 0xd3, 0xd4, 42);
  TPRINTF("d1=%04x d2=%06x d3=%08x %%100\n", 0xd1, 0xd2, 0xd3);
  TPRINTF("|%-14s| |%-16s| d2=%2x |%-30s|\n", "str14", "str16", 0xd2,
          "12345678901234567890123456789012345");
  TPRINTF("|%4s|\n", "string4");
  TPRINTF("|%-4s|\n", "string4");
  TPRINTF("42=%3d d1=%4.4x |%4s| d2=%8.8x\n", 42, 0xd1, "string4", 0xd2);
  TPRINTF("42=%3d d1=%4.4x |%-4s| d2=%8.8x\n", 42, 0xd1, "string4", 0xd2);
  TPRINTF("84=%d 21=%ds |%s| |%sOK| d1=%x d2=%#x\n",
          84, 21, "hello", "fine", 0xd1, 0xd2);
  TPRINTF("42=% 3d d1=%4x |%10s| d2=%3.3x\n", 42, 0xd1, "string4", 0xd2);

  TPRINTF("%lld\n", LLONG_MIN);
  TPRINTF("%lld\n", LLONG_MAX);
  TPRINTF("%llu\n", ULLONG_MAX);
  TPRINTF("%llx\n", LLONG_MIN);
  TPRINTF("%llx\n", LLONG_MAX);
  TPRINTF("%llx\n", ULLONG_MAX);

  TPRINTF("d1=%.1x\n", 0xd1);
  TPRINTF("d1=%4.1x\n", 0xd1);
  TPRINTF("d1=%4.x\n", 0xd1);

  {
    char blah[256];
    TPRINTF("a=%zd\n", sizeof(blah));
    TPRINTF("a=%zu\n", sizeof(blah));
    TPRINTF("a=%zi\n", sizeof(blah));
    TPRINTF("a=0x%zx\n", sizeof(blah));
  }

  {
    int in_stack;
    TPRINTF("Adddress of main: %p\n", main);
    TPRINTF("Adddress of stack variable: %p\n", &in_stack);
  }

  return 0;
}
