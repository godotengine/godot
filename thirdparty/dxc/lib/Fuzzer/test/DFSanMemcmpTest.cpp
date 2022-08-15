// Simple test for a fuzzer. The fuzzer must find a particular string.
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size >= 8 && memcmp(Data, "01234567", 8) == 0) {
    fprintf(stderr, "BINGO\n");
    exit(1);
  }
}
