//=-- InstrProfWriter.cpp - Instrumented profiling writer -------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfWriter.h"
#include "InstrProfIndexed.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/OnDiskHashTable.h"

using namespace llvm;

namespace {
class InstrProfRecordTrait {
public:
  typedef StringRef key_type;
  typedef StringRef key_type_ref;

  typedef const InstrProfWriter::CounterData *const data_type;
  typedef const InstrProfWriter::CounterData *const data_type_ref;

  typedef uint64_t hash_value_type;
  typedef uint64_t offset_type;

  static hash_value_type ComputeHash(key_type_ref K) {
    return IndexedInstrProf::ComputeHash(IndexedInstrProf::HashType, K);
  }

  static std::pair<offset_type, offset_type>
  EmitKeyDataLength(raw_ostream &Out, key_type_ref K, data_type_ref V) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);

    offset_type N = K.size();
    LE.write<offset_type>(N);

    offset_type M = 0;
    for (const auto &Counts : *V)
      M += (2 + Counts.second.size()) * sizeof(uint64_t);
    LE.write<offset_type>(M);

    return std::make_pair(N, M);
  }

  static void EmitKey(raw_ostream &Out, key_type_ref K, offset_type N){
    Out.write(K.data(), N);
  }

  static void EmitData(raw_ostream &Out, key_type_ref, data_type_ref V,
                       offset_type) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);

    for (const auto &Counts : *V) {
      LE.write<uint64_t>(Counts.first);
      LE.write<uint64_t>(Counts.second.size());
      for (uint64_t I : Counts.second)
        LE.write<uint64_t>(I);
    }
  }
};
}

std::error_code
InstrProfWriter::addFunctionCounts(StringRef FunctionName,
                                   uint64_t FunctionHash,
                                   ArrayRef<uint64_t> Counters) {
  auto &CounterData = FunctionData[FunctionName];

  auto Where = CounterData.find(FunctionHash);
  if (Where == CounterData.end()) {
    // We've never seen a function with this name and hash, add it.
    CounterData[FunctionHash] = Counters;
    // We keep track of the max function count as we go for simplicity.
    if (Counters[0] > MaxFunctionCount)
      MaxFunctionCount = Counters[0];
    return instrprof_error::success;
  }

  // We're updating a function we've seen before.
  auto &FoundCounters = Where->second;
  // If the number of counters doesn't match we either have bad data or a hash
  // collision.
  if (FoundCounters.size() != Counters.size())
    return instrprof_error::count_mismatch;

  for (size_t I = 0, E = Counters.size(); I < E; ++I) {
    if (FoundCounters[I] + Counters[I] < FoundCounters[I])
      return instrprof_error::counter_overflow;
    FoundCounters[I] += Counters[I];
  }
  // We keep track of the max function count as we go for simplicity.
  if (FoundCounters[0] > MaxFunctionCount)
    MaxFunctionCount = FoundCounters[0];

  return instrprof_error::success;
}

std::pair<uint64_t, uint64_t> InstrProfWriter::writeImpl(raw_ostream &OS) {
  OnDiskChainedHashTableGenerator<InstrProfRecordTrait> Generator;

  // Populate the hash table generator.
  for (const auto &I : FunctionData)
    Generator.insert(I.getKey(), &I.getValue());

  using namespace llvm::support;
  endian::Writer<little> LE(OS);

  // Write the header.
  LE.write<uint64_t>(IndexedInstrProf::Magic);
  LE.write<uint64_t>(IndexedInstrProf::Version);
  LE.write<uint64_t>(MaxFunctionCount);
  LE.write<uint64_t>(static_cast<uint64_t>(IndexedInstrProf::HashType));

  // Save a space to write the hash table start location.
  uint64_t HashTableStartLoc = OS.tell();
  LE.write<uint64_t>(0);
  // Write the hash table.
  uint64_t HashTableStart = Generator.Emit(OS);

  return std::make_pair(HashTableStartLoc, HashTableStart);
}

void InstrProfWriter::write(raw_fd_ostream &OS) {
  // Write the hash table.
  auto TableStart = writeImpl(OS);

  // Go back and fill in the hash table start.
  using namespace support;
  OS.seek(TableStart.first);
  endian::Writer<little>(OS).write<uint64_t>(TableStart.second);
}

std::unique_ptr<MemoryBuffer> InstrProfWriter::writeBuffer() {
  std::string Data;
  llvm::raw_string_ostream OS(Data);
  // Write the hash table.
  auto TableStart = writeImpl(OS);
  OS.flush();

  // Go back and fill in the hash table start.
  using namespace support;
  uint64_t Bytes = endian::byte_swap<uint64_t, little>(TableStart.second);
  Data.replace(TableStart.first, sizeof(uint64_t), (const char *)&Bytes,
               sizeof(uint64_t));

  // Return this in an aligned memory buffer.
  return MemoryBuffer::getMemBufferCopy(Data);
}
