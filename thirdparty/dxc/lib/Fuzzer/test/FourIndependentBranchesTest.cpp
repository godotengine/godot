// Simple test for a fuzzer. The fuzzer must find the string "FUZZ".
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <iostream>

extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  int bits = 0;
  if (Size > 0 && Data[0] == 'F') bits |= 1;
  if (Size > 1 && Data[1] == 'U') bits |= 2;
  if (Size > 2 && Data[2] == 'Z') bits |= 4;
  if (Size > 3 && Data[3] == 'Z') bits |= 8;
  if (bits == 15) {
    std::cerr <<  "BINGO!\n";
    exit(1);
  }
}

