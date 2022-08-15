//===- FuzzerInterface.h - Interface header for the Fuzzer ------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Define the interface between the Fuzzer and the library being tested.
//===----------------------------------------------------------------------===//

// WARNING: keep the interface free of STL or any other header-based C++ lib,
// to avoid bad interactions between the code used in the fuzzer and
// the code used in the target function.

#ifndef LLVM_FUZZER_INTERFACE_H
#define LLVM_FUZZER_INTERFACE_H

#include <cstddef>
#include <cstdint>

namespace fuzzer {

typedef void (*UserCallback)(const uint8_t *Data, size_t Size);
/** Simple C-like interface with a single user-supplied callback.

Usage:

#\code
#include "FuzzerInterface.h"

void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  DoStuffWithData(Data, Size);
}

// Implement your own main() or use the one from FuzzerMain.cpp.
int main(int argc, char **argv) {
  InitializeMeIfNeeded();
  return fuzzer::FuzzerDriver(argc, argv, LLVMFuzzerTestOneInput);
}
#\endcode
*/
int FuzzerDriver(int argc, char **argv, UserCallback Callback);

/** An abstract class that allows to use user-supplied mutators with libFuzzer.

Usage:

#\code
#include "FuzzerInterface.h"
class MyFuzzer : public fuzzer::UserSuppliedFuzzer {
 public:
  // Must define the target function.
  void TargetFunction(...) { ... }
  // Optionally define the mutator.
  size_t Mutate(...) { ... }
  // Optionally define the CrossOver method.
  size_t CrossOver(...) { ... }
};

int main(int argc, char **argv) {
  MyFuzzer F;
  fuzzer::FuzzerDriver(argc, argv, F);
}
#\endcode
*/
class UserSuppliedFuzzer {
 public:
  /// Executes the target function on 'Size' bytes of 'Data'.
  virtual void TargetFunction(const uint8_t *Data, size_t Size) = 0;
  /// Mutates 'Size' bytes of data in 'Data' inplace into up to 'MaxSize' bytes,
  /// returns the new size of the data, which should be positive.
  virtual size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize) {
    return BasicMutate(Data, Size, MaxSize);
  }
  /// Crosses 'Data1' and 'Data2', writes up to 'MaxOutSize' bytes into Out,
  /// returns the number of bytes written, which should be positive.
  virtual size_t CrossOver(const uint8_t *Data1, size_t Size1,
                           const uint8_t *Data2, size_t Size2,
                           uint8_t *Out, size_t MaxOutSize) {
    return BasicCrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize);
  }
  virtual ~UserSuppliedFuzzer() {}

 protected:
  /// These can be called internally by Mutate and CrossOver.
  size_t BasicMutate(uint8_t *Data, size_t Size, size_t MaxSize);
  size_t BasicCrossOver(const uint8_t *Data1, size_t Size1,
                        const uint8_t *Data2, size_t Size2,
                        uint8_t *Out, size_t MaxOutSize);
};

/// Runs the fuzzing with the UserSuppliedFuzzer.
int FuzzerDriver(int argc, char **argv, UserSuppliedFuzzer &USF);

}  // namespace fuzzer

#endif  // LLVM_FUZZER_INTERFACE_H
