//=-- InstrProfWriter.h - Instrumented profiling writer -----------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for instrumentation
// based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_INSTRPROFWRITER_H
#define LLVM_PROFILEDATA_INSTRPROFWRITER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace llvm {

/// Writer for instrumentation based profile data.
class InstrProfWriter {
public:
  typedef SmallDenseMap<uint64_t, std::vector<uint64_t>, 1> CounterData;
private:
  StringMap<CounterData> FunctionData;
  uint64_t MaxFunctionCount;
public:
  InstrProfWriter() : MaxFunctionCount(0) {}

  /// Add function counts for the given function. If there are already counts
  /// for this function and the hash and number of counts match, each counter is
  /// summed.
  std::error_code addFunctionCounts(StringRef FunctionName,
                                    uint64_t FunctionHash,
                                    ArrayRef<uint64_t> Counters);
  /// Write the profile to \c OS
  void write(raw_fd_ostream &OS);
  /// Write the profile, returning the raw data. For testing.
  std::unique_ptr<MemoryBuffer> writeBuffer();

private:
  std::pair<uint64_t, uint64_t> writeImpl(raw_ostream &OS);
};

} // end namespace llvm

#endif
