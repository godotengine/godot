//=-- InstrProfReader.cpp - Instrumented profiling reader -------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfReader.h"
#include "InstrProfIndexed.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

using namespace llvm;

static ErrorOr<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(std::string Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = BufferOrErr.getError())
    return EC;
  return std::move(BufferOrErr.get());
}

static std::error_code initializeReader(InstrProfReader &Reader) {
  return Reader.readHeader();
}

ErrorOr<std::unique_ptr<InstrProfReader>>
InstrProfReader::create(std::string Path) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path);
  if (std::error_code EC = BufferOrError.getError())
    return EC;
  return InstrProfReader::create(std::move(BufferOrError.get()));
}

ErrorOr<std::unique_ptr<InstrProfReader>>
InstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer) {
  // Sanity check the buffer.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return instrprof_error::too_large;

  std::unique_ptr<InstrProfReader> Result;
  // Create the reader.
  if (IndexedInstrProfReader::hasFormat(*Buffer))
    Result.reset(new IndexedInstrProfReader(std::move(Buffer)));
  else if (RawInstrProfReader64::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader64(std::move(Buffer)));
  else if (RawInstrProfReader32::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader32(std::move(Buffer)));
  else
    Result.reset(new TextInstrProfReader(std::move(Buffer)));

  // Initialize the reader and return the result.
  if (std::error_code EC = initializeReader(*Result))
    return EC;

  return std::move(Result);
}

ErrorOr<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(std::string Path) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path);
  if (std::error_code EC = BufferOrError.getError())
    return EC;
  return IndexedInstrProfReader::create(std::move(BufferOrError.get()));
}


ErrorOr<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer) {
  // Sanity check the buffer.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return instrprof_error::too_large;

  // Create the reader.
  if (!IndexedInstrProfReader::hasFormat(*Buffer))
    return instrprof_error::bad_magic;
  auto Result = llvm::make_unique<IndexedInstrProfReader>(std::move(Buffer));

  // Initialize the reader and return the result.
  if (std::error_code EC = initializeReader(*Result))
    return EC;

  return std::move(Result);
}

void InstrProfIterator::Increment() {
  if (Reader->readNextRecord(Record))
    *this = InstrProfIterator();
}

std::error_code TextInstrProfReader::readNextRecord(InstrProfRecord &Record) {
  // Skip empty lines and comments.
  while (!Line.is_at_end() && (Line->empty() || Line->startswith("#")))
    ++Line;
  // If we hit EOF while looking for a name, we're done.
  if (Line.is_at_end())
    return error(instrprof_error::eof);

  // Read the function name.
  Record.Name = *Line++;

  // Read the function hash.
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(0, Record.Hash))
    return error(instrprof_error::malformed);

  // Read the number of counters.
  uint64_t NumCounters;
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(10, NumCounters))
    return error(instrprof_error::malformed);
  if (NumCounters == 0)
    return error(instrprof_error::malformed);

  // Read each counter and fill our internal storage with the values.
  Record.Counts.clear();
  Record.Counts.reserve(NumCounters);
  for (uint64_t I = 0; I < NumCounters; ++I) {
    if (Line.is_at_end())
      return error(instrprof_error::truncated);
    uint64_t Count;
    if ((Line++)->getAsInteger(10, Count))
      return error(instrprof_error::malformed);
    Record.Counts.push_back(Count);
  }

  return success();
}

template <class IntPtrT>
static uint64_t getRawMagic();

template <>
uint64_t getRawMagic<uint64_t>() {
  return
    uint64_t(255) << 56 |
    uint64_t('l') << 48 |
    uint64_t('p') << 40 |
    uint64_t('r') << 32 |
    uint64_t('o') << 24 |
    uint64_t('f') << 16 |
    uint64_t('r') <<  8 |
    uint64_t(129);
}

template <>
uint64_t getRawMagic<uint32_t>() {
  return
    uint64_t(255) << 56 |
    uint64_t('l') << 48 |
    uint64_t('p') << 40 |
    uint64_t('r') << 32 |
    uint64_t('o') << 24 |
    uint64_t('f') << 16 |
    uint64_t('R') <<  8 |
    uint64_t(129);
}

template <class IntPtrT>
bool RawInstrProfReader<IntPtrT>::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < sizeof(uint64_t))
    return false;
  uint64_t Magic =
    *reinterpret_cast<const uint64_t *>(DataBuffer.getBufferStart());
  return getRawMagic<IntPtrT>() == Magic ||
    sys::getSwappedBytes(getRawMagic<IntPtrT>()) == Magic;
}

template <class IntPtrT>
std::error_code RawInstrProfReader<IntPtrT>::readHeader() {
  if (!hasFormat(*DataBuffer))
    return error(instrprof_error::bad_magic);
  if (DataBuffer->getBufferSize() < sizeof(RawHeader))
    return error(instrprof_error::bad_header);
  auto *Header =
    reinterpret_cast<const RawHeader *>(DataBuffer->getBufferStart());
  ShouldSwapBytes = Header->Magic != getRawMagic<IntPtrT>();
  return readHeader(*Header);
}

template <class IntPtrT>
std::error_code
RawInstrProfReader<IntPtrT>::readNextHeader(const char *CurrentPos) {
  const char *End = DataBuffer->getBufferEnd();
  // Skip zero padding between profiles.
  while (CurrentPos != End && *CurrentPos == 0)
    ++CurrentPos;
  // If there's nothing left, we're done.
  if (CurrentPos == End)
    return instrprof_error::eof;
  // If there isn't enough space for another header, this is probably just
  // garbage at the end of the file.
  if (CurrentPos + sizeof(RawHeader) > End)
    return instrprof_error::malformed;
  // The writer ensures each profile is padded to start at an aligned address.
  if (reinterpret_cast<size_t>(CurrentPos) % alignOf<uint64_t>())
    return instrprof_error::malformed;
  // The magic should have the same byte order as in the previous header.
  uint64_t Magic = *reinterpret_cast<const uint64_t *>(CurrentPos);
  if (Magic != swap(getRawMagic<IntPtrT>()))
    return instrprof_error::bad_magic;

  // There's another profile to read, so we need to process the header.
  auto *Header = reinterpret_cast<const RawHeader *>(CurrentPos);
  return readHeader(*Header);
}

static uint64_t getRawVersion() {
  return 1;
}

template <class IntPtrT>
std::error_code
RawInstrProfReader<IntPtrT>::readHeader(const RawHeader &Header) {
  if (swap(Header.Version) != getRawVersion())
    return error(instrprof_error::unsupported_version);

  CountersDelta = swap(Header.CountersDelta);
  NamesDelta = swap(Header.NamesDelta);
  auto DataSize = swap(Header.DataSize);
  auto CountersSize = swap(Header.CountersSize);
  auto NamesSize = swap(Header.NamesSize);

  ptrdiff_t DataOffset = sizeof(RawHeader);
  ptrdiff_t CountersOffset = DataOffset + sizeof(ProfileData) * DataSize;
  ptrdiff_t NamesOffset = CountersOffset + sizeof(uint64_t) * CountersSize;
  size_t ProfileSize = NamesOffset + sizeof(char) * NamesSize;

  auto *Start = reinterpret_cast<const char *>(&Header);
  if (Start + ProfileSize > DataBuffer->getBufferEnd())
    return error(instrprof_error::bad_header);

  Data = reinterpret_cast<const ProfileData *>(Start + DataOffset);
  DataEnd = Data + DataSize;
  CountersStart = reinterpret_cast<const uint64_t *>(Start + CountersOffset);
  NamesStart = Start + NamesOffset;
  ProfileEnd = Start + ProfileSize;

  return success();
}

template <class IntPtrT>
std::error_code
RawInstrProfReader<IntPtrT>::readNextRecord(InstrProfRecord &Record) {
  if (Data == DataEnd)
    if (std::error_code EC = readNextHeader(ProfileEnd))
      return EC;

  // Get the raw data.
  StringRef RawName(getName(Data->NamePtr), swap(Data->NameSize));
  uint32_t NumCounters = swap(Data->NumCounters);
  if (NumCounters == 0)
    return error(instrprof_error::malformed);
  auto RawCounts = makeArrayRef(getCounter(Data->CounterPtr), NumCounters);

  // Check bounds.
  auto *NamesStartAsCounter = reinterpret_cast<const uint64_t *>(NamesStart);
  if (RawName.data() < NamesStart ||
      RawName.data() + RawName.size() > DataBuffer->getBufferEnd() ||
      RawCounts.data() < CountersStart ||
      RawCounts.data() + RawCounts.size() > NamesStartAsCounter)
    return error(instrprof_error::malformed);

  // Store the data in Record, byte-swapping as necessary.
  Record.Hash = swap(Data->FuncHash);
  Record.Name = RawName;
  if (ShouldSwapBytes) {
    Record.Counts.clear();
    Record.Counts.reserve(RawCounts.size());
    for (uint64_t Count : RawCounts)
      Record.Counts.push_back(swap(Count));
  } else
    Record.Counts = RawCounts;

  // Iterate.
  ++Data;
  return success();
}

namespace llvm {
template class RawInstrProfReader<uint32_t>;
template class RawInstrProfReader<uint64_t>;
}

InstrProfLookupTrait::hash_value_type
InstrProfLookupTrait::ComputeHash(StringRef K) {
  return IndexedInstrProf::ComputeHash(HashType, K);
}

typedef InstrProfLookupTrait::data_type data_type;
typedef InstrProfLookupTrait::offset_type offset_type;

data_type InstrProfLookupTrait::ReadData(StringRef K, const unsigned char *D,
                                         offset_type N) {

  // Check if the data is corrupt. If so, don't try to read it.
  if (N % sizeof(uint64_t))
    return data_type();

  DataBuffer.clear();
  uint64_t NumCounts;
  uint64_t NumEntries = N / sizeof(uint64_t);
  std::vector<uint64_t> CounterBuffer;
  for (uint64_t I = 0; I < NumEntries; I += NumCounts) {
    using namespace support;
    // The function hash comes first.
    uint64_t Hash = endian::readNext<uint64_t, little, unaligned>(D);

    if (++I >= NumEntries)
      return data_type();

    // In v1, we have at least one count.
    // Later, we have the number of counts.
    NumCounts = (1 == FormatVersion)
                    ? NumEntries - I
                    : endian::readNext<uint64_t, little, unaligned>(D);
    if (1 != FormatVersion)
      ++I;

    // If we have more counts than data, this is bogus.
    if (I + NumCounts > NumEntries)
      return data_type();

    CounterBuffer.clear();
    for (unsigned J = 0; J < NumCounts; ++J)
      CounterBuffer.push_back(endian::readNext<uint64_t, little, unaligned>(D));

    DataBuffer.push_back(InstrProfRecord(K, Hash, std::move(CounterBuffer)));
  }
  return DataBuffer;
}

bool IndexedInstrProfReader::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < 8)
    return false;
  using namespace support;
  uint64_t Magic =
      endian::read<uint64_t, little, aligned>(DataBuffer.getBufferStart());
  return Magic == IndexedInstrProf::Magic;
}

std::error_code IndexedInstrProfReader::readHeader() {
  const unsigned char *Start =
      (const unsigned char *)DataBuffer->getBufferStart();
  const unsigned char *Cur = Start;
  if ((const unsigned char *)DataBuffer->getBufferEnd() - Cur < 24)
    return error(instrprof_error::truncated);

  using namespace support;

  // Check the magic number.
  uint64_t Magic = endian::readNext<uint64_t, little, unaligned>(Cur);
  if (Magic != IndexedInstrProf::Magic)
    return error(instrprof_error::bad_magic);

  // Read the version.
  FormatVersion = endian::readNext<uint64_t, little, unaligned>(Cur);
  if (FormatVersion > IndexedInstrProf::Version)
    return error(instrprof_error::unsupported_version);

  // Read the maximal function count.
  MaxFunctionCount = endian::readNext<uint64_t, little, unaligned>(Cur);

  // Read the hash type and start offset.
  IndexedInstrProf::HashT HashType = static_cast<IndexedInstrProf::HashT>(
      endian::readNext<uint64_t, little, unaligned>(Cur));
  if (HashType > IndexedInstrProf::HashT::Last)
    return error(instrprof_error::unsupported_hash_type);
  uint64_t HashOffset = endian::readNext<uint64_t, little, unaligned>(Cur);

  // The rest of the file is an on disk hash table.
  Index.reset(InstrProfReaderIndex::Create(
      Start + HashOffset, Cur, Start,
      InstrProfLookupTrait(HashType, FormatVersion)));
  // Set up our iterator for readNextRecord.
  RecordIterator = Index->data_begin();

  return success();
}

std::error_code IndexedInstrProfReader::getFunctionCounts(
    StringRef FuncName, uint64_t FuncHash, std::vector<uint64_t> &Counts) {
  auto Iter = Index->find(FuncName);
  if (Iter == Index->end())
    return error(instrprof_error::unknown_function);

  // Found it. Look for counters with the right hash.
  ArrayRef<InstrProfRecord> Data = (*Iter);
  if (Data.empty())
    return error(instrprof_error::malformed);

  for (unsigned I = 0, E = Data.size(); I < E; ++I) {
    // Check for a match and fill the vector if there is one.
    if (Data[I].Hash == FuncHash) {
      Counts = Data[I].Counts;
      return success();
    }
  }
  return error(instrprof_error::hash_mismatch);
}

std::error_code
IndexedInstrProfReader::readNextRecord(InstrProfRecord &Record) {
  // Are we out of records?
  if (RecordIterator == Index->data_end())
    return error(instrprof_error::eof);

  if ((*RecordIterator).empty())
    return error(instrprof_error::malformed);

  static unsigned RecordIndex = 0;
  ArrayRef<InstrProfRecord> Data = (*RecordIterator);
  Record = Data[RecordIndex++];
  if (RecordIndex >= Data.size()) {
    ++RecordIterator;
    RecordIndex = 0;
  }
  return success();
}
