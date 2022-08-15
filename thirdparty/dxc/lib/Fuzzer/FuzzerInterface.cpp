//===- FuzzerInterface.cpp - Mutate a test input --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Parts of public interface for libFuzzer.
//===----------------------------------------------------------------------===//

#include "FuzzerInterface.h"
#include "FuzzerInternal.h"

namespace fuzzer {
size_t UserSuppliedFuzzer::BasicMutate(uint8_t *Data, size_t Size,
                                       size_t MaxSize) {
  return ::fuzzer::Mutate(Data, Size, MaxSize);
}
size_t UserSuppliedFuzzer::BasicCrossOver(const uint8_t *Data1, size_t Size1,
                                          const uint8_t *Data2, size_t Size2,
                                          uint8_t *Out, size_t MaxOutSize) {
  return ::fuzzer::CrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize);
}

}  // namespace fuzzer.
