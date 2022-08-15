// Simple test for a fuzzer. The fuzzer must find the string "Hi!".

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <iostream>

static volatile int Sink;
static volatile int *Null = 0;

extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size > 0 && Data[0] == 'H') {
    Sink = 1;
    if (Size > 1 && Data[1] == 'i') {
      Sink = 2;
      if (Size > 2 && Data[2] == '!') {
        std::cout << "Found the target, dereferencing NULL\n";
        *Null = 1;
      }
    }
  }
}

