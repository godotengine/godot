// Simple test for a fuzzer. The fuzzer must find several narrow ranges.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size < 14) return;
  uint64_t x = 0;
  int64_t  y = 0;
  int z = 0;
  unsigned short a = 0;
  memcpy(&x, Data, 8);
  memcpy(&y, Data + Size - 8, 8);
  memcpy(&z, Data + Size / 2, sizeof(z));
  memcpy(&a, Data + Size / 2 + 4, sizeof(a));

  if (x > 1234567890 &&
      x < 1234567895 &&
      y >= 987654321 &&
      y <= 987654325 &&
      z < -10000 &&
      z >= -10005 &&
      z != -10003 &&
      a == 4242) {
    fprintf(stderr, "BINGO; Found the target: size %zd (%zd, %zd, %d, %d), exiting.\n",
            Size, x, y, z, a);
    exit(1);
  }
}
