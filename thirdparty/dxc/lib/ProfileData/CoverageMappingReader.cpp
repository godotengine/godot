//=-- CoverageMappingReader.cpp - Code coverage mapping reader ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading coverage mapping data for
// instrumentation based coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace coverage;
using namespace object;

#define DEBUG_TYPE "coverage-mapping"

void CoverageMappingIterator::increment() {
  // Check if all the records were read or if an error occurred while reading
  // the next record.
  if (Reader->readNextRecord(Record))
    *this = CoverageMappingIterator();
}

std::error_code RawCoverageReader::readULEB128(uint64_t &Result) {
  if (Data.size() < 1)
    return coveragemap_error::truncated;
  unsigned N = 0;
  Result = decodeULEB128(reinterpret_cast<const uint8_t *>(Data.data()), &N);
  if (N > Data.size())
    return coveragemap_error::malformed;
  Data = Data.substr(N);
  return std::error_code();
}

std::error_code RawCoverageReader::readIntMax(uint64_t &Result,
                                              uint64_t MaxPlus1) {
  if (auto Err = readULEB128(Result))
    return Err;
  if (Result >= MaxPlus1)
    return coveragemap_error::malformed;
  return std::error_code();
}

std::error_code RawCoverageReader::readSize(uint64_t &Result) {
  if (auto Err = readULEB128(Result))
    return Err;
  // Sanity check the number.
  if (Result > Data.size())
    return coveragemap_error::malformed;
  return std::error_code();
}

std::error_code RawCoverageReader::readString(StringRef &Result) {
  uint64_t Length;
  if (auto Err = readSize(Length))
    return Err;
  Result = Data.substr(0, Length);
  Data = Data.substr(Length);
  return std::error_code();
}

std::error_code RawCoverageFilenamesReader::read() {
  uint64_t NumFilenames;
  if (auto Err = readSize(NumFilenames))
    return Err;
  for (size_t I = 0; I < NumFilenames; ++I) {
    StringRef Filename;
    if (auto Err = readString(Filename))
      return Err;
    Filenames.push_back(Filename);
  }
  return std::error_code();
}

std::error_code RawCoverageMappingReader::decodeCounter(unsigned Value,
                                                        Counter &C) {
  auto Tag = Value & Counter::EncodingTagMask;
  switch (Tag) {
  case Counter::Zero:
    C = Counter::getZero();
    return std::error_code();
  case Counter::CounterValueReference:
    C = Counter::getCounter(Value >> Counter::EncodingTagBits);
    return std::error_code();
  default:
    break;
  }
  Tag -= Counter::Expression;
  switch (Tag) {
  case CounterExpression::Subtract:
  case CounterExpression::Add: {
    auto ID = Value >> Counter::EncodingTagBits;
    if (ID >= Expressions.size())
      return coveragemap_error::malformed;
    Expressions[ID].Kind = CounterExpression::ExprKind(Tag);
    C = Counter::getExpression(ID);
    break;
  }
  default:
    return coveragemap_error::malformed;
  }
  return std::error_code();
}

std::error_code RawCoverageMappingReader::readCounter(Counter &C) {
  uint64_t EncodedCounter;
  if (auto Err =
          readIntMax(EncodedCounter, std::numeric_limits<unsigned>::max()))
    return Err;
  if (auto Err = decodeCounter(EncodedCounter, C))
    return Err;
  return std::error_code();
}

static const unsigned EncodingExpansionRegionBit = 1
                                                   << Counter::EncodingTagBits;

/// \brief Read the sub-array of regions for the given inferred file id.
/// \param NumFileIDs the number of file ids that are defined for this
/// function.
std::error_code RawCoverageMappingReader::readMappingRegionsSubArray(
    std::vector<CounterMappingRegion> &MappingRegions, unsigned InferredFileID,
    size_t NumFileIDs) {
  uint64_t NumRegions;
  if (auto Err = readSize(NumRegions))
    return Err;
  unsigned LineStart = 0;
  for (size_t I = 0; I < NumRegions; ++I) {
    Counter C;
    CounterMappingRegion::RegionKind Kind = CounterMappingRegion::CodeRegion;

    // Read the combined counter + region kind.
    uint64_t EncodedCounterAndRegion;
    if (auto Err = readIntMax(EncodedCounterAndRegion,
                              std::numeric_limits<unsigned>::max()))
      return Err;
    unsigned Tag = EncodedCounterAndRegion & Counter::EncodingTagMask;
    uint64_t ExpandedFileID = 0;
    if (Tag != Counter::Zero) {
      if (auto Err = decodeCounter(EncodedCounterAndRegion, C))
        return Err;
    } else {
      // Is it an expansion region?
      if (EncodedCounterAndRegion & EncodingExpansionRegionBit) {
        Kind = CounterMappingRegion::ExpansionRegion;
        ExpandedFileID = EncodedCounterAndRegion >>
                         Counter::EncodingCounterTagAndExpansionRegionTagBits;
        if (ExpandedFileID >= NumFileIDs)
          return coveragemap_error::malformed;
      } else {
        switch (EncodedCounterAndRegion >>
                Counter::EncodingCounterTagAndExpansionRegionTagBits) {
        case CounterMappingRegion::CodeRegion:
          // Don't do anything when we have a code region with a zero counter.
          break;
        case CounterMappingRegion::SkippedRegion:
          Kind = CounterMappingRegion::SkippedRegion;
          break;
        default:
          return coveragemap_error::malformed;
        }
      }
    }

    // Read the source range.
    uint64_t LineStartDelta, ColumnStart, NumLines, ColumnEnd;
    if (auto Err =
            readIntMax(LineStartDelta, std::numeric_limits<unsigned>::max()))
      return Err;
    if (auto Err = readULEB128(ColumnStart))
      return Err;
    if (ColumnStart > std::numeric_limits<unsigned>::max())
      return coveragemap_error::malformed;
    if (auto Err = readIntMax(NumLines, std::numeric_limits<unsigned>::max()))
      return Err;
    if (auto Err = readIntMax(ColumnEnd, std::numeric_limits<unsigned>::max()))
      return Err;
    LineStart += LineStartDelta;
    // Adjust the column locations for the empty regions that are supposed to
    // cover whole lines. Those regions should be encoded with the
    // column range (1 -> std::numeric_limits<unsigned>::max()), but because
    // the encoded std::numeric_limits<unsigned>::max() is several bytes long,
    // we set the column range to (0 -> 0) to ensure that the column start and
    // column end take up one byte each.
    // The std::numeric_limits<unsigned>::max() is used to represent a column
    // position at the end of the line without knowing the length of that line.
    if (ColumnStart == 0 && ColumnEnd == 0) {
      ColumnStart = 1;
      ColumnEnd = std::numeric_limits<unsigned>::max();
    }

    DEBUG({
      dbgs() << "Counter in file " << InferredFileID << " " << LineStart << ":"
             << ColumnStart << " -> " << (LineStart + NumLines) << ":"
             << ColumnEnd << ", ";
      if (Kind == CounterMappingRegion::ExpansionRegion)
        dbgs() << "Expands to file " << ExpandedFileID;
      else
        CounterMappingContext(Expressions).dump(C, dbgs());
      dbgs() << "\n";
    });

    MappingRegions.push_back(CounterMappingRegion(
        C, InferredFileID, ExpandedFileID, LineStart, ColumnStart,
        LineStart + NumLines, ColumnEnd, Kind));
  }
  return std::error_code();
}

std::error_code RawCoverageMappingReader::read() {

  // Read the virtual file mapping.
  llvm::SmallVector<unsigned, 8> VirtualFileMapping;
  uint64_t NumFileMappings;
  if (auto Err = readSize(NumFileMappings))
    return Err;
  for (size_t I = 0; I < NumFileMappings; ++I) {
    uint64_t FilenameIndex;
    if (auto Err = readIntMax(FilenameIndex, TranslationUnitFilenames.size()))
      return Err;
    VirtualFileMapping.push_back(FilenameIndex);
  }

  // Construct the files using unique filenames and virtual file mapping.
  for (auto I : VirtualFileMapping) {
    Filenames.push_back(TranslationUnitFilenames[I]);
  }

  // Read the expressions.
  uint64_t NumExpressions;
  if (auto Err = readSize(NumExpressions))
    return Err;
  // Create an array of dummy expressions that get the proper counters
  // when the expressions are read, and the proper kinds when the counters
  // are decoded.
  Expressions.resize(
      NumExpressions,
      CounterExpression(CounterExpression::Subtract, Counter(), Counter()));
  for (size_t I = 0; I < NumExpressions; ++I) {
    if (auto Err = readCounter(Expressions[I].LHS))
      return Err;
    if (auto Err = readCounter(Expressions[I].RHS))
      return Err;
  }

  // Read the mapping regions sub-arrays.
  for (unsigned InferredFileID = 0, S = VirtualFileMapping.size();
       InferredFileID < S; ++InferredFileID) {
    if (auto Err = readMappingRegionsSubArray(MappingRegions, InferredFileID,
                                              VirtualFileMapping.size()))
      return Err;
  }

  // Set the counters for the expansion regions.
  // i.e. Counter of expansion region = counter of the first region
  // from the expanded file.
  // Perform multiple passes to correctly propagate the counters through
  // all the nested expansion regions.
  SmallVector<CounterMappingRegion *, 8> FileIDExpansionRegionMapping;
  FileIDExpansionRegionMapping.resize(VirtualFileMapping.size(), nullptr);
  for (unsigned Pass = 1, S = VirtualFileMapping.size(); Pass < S; ++Pass) {
    for (auto &R : MappingRegions) {
      if (R.Kind != CounterMappingRegion::ExpansionRegion)
        continue;
      assert(!FileIDExpansionRegionMapping[R.ExpandedFileID]);
      FileIDExpansionRegionMapping[R.ExpandedFileID] = &R;
    }
    for (auto &R : MappingRegions) {
      if (FileIDExpansionRegionMapping[R.FileID]) {
        FileIDExpansionRegionMapping[R.FileID]->Count = R.Count;
        FileIDExpansionRegionMapping[R.FileID] = nullptr;
      }
    }
  }

  return std::error_code();
}

namespace {

/// \brief A helper structure to access the data from a section
/// in an object file.
struct SectionData {
  StringRef Data;
  uint64_t Address;

  std::error_code load(SectionRef &Section) {
    if (auto Err = Section.getContents(Data))
      return Err;
    Address = Section.getAddress();
    return std::error_code();
  }

  std::error_code get(uint64_t Pointer, size_t Size, StringRef &Result) {
    if (Pointer < Address)
      return coveragemap_error::malformed;
    auto Offset = Pointer - Address;
    if (Offset + Size > Data.size())
      return coveragemap_error::malformed;
    Result = Data.substr(Pointer - Address, Size);
    return std::error_code();
  }
};
}

template <typename T, support::endianness Endian>
std::error_code readCoverageMappingData(
    SectionData &ProfileNames, StringRef Data,
    std::vector<BinaryCoverageReader::ProfileMappingRecord> &Records,
    std::vector<StringRef> &Filenames) {
  using namespace support;
  llvm::DenseSet<T> UniqueFunctionMappingData;

  // Read the records in the coverage data section.
  for (const char *Buf = Data.data(), *End = Buf + Data.size(); Buf < End;) {
    if (Buf + 4 * sizeof(uint32_t) > End)
      return coveragemap_error::malformed;
    uint32_t NRecords = endian::readNext<uint32_t, Endian, unaligned>(Buf);
    uint32_t FilenamesSize = endian::readNext<uint32_t, Endian, unaligned>(Buf);
    uint32_t CoverageSize = endian::readNext<uint32_t, Endian, unaligned>(Buf);
    uint32_t Version = endian::readNext<uint32_t, Endian, unaligned>(Buf);

    switch (Version) {
    case CoverageMappingVersion1:
      break;
    default:
      return coveragemap_error::unsupported_version;
    }

    // Skip past the function records, saving the start and end for later.
    const char *FunBuf = Buf;
    Buf += NRecords * (sizeof(T) + 2 * sizeof(uint32_t) + sizeof(uint64_t));
    const char *FunEnd = Buf;

    // Get the filenames.
    if (Buf + FilenamesSize > End)
      return coveragemap_error::malformed;
    size_t FilenamesBegin = Filenames.size();
    RawCoverageFilenamesReader Reader(StringRef(Buf, FilenamesSize), Filenames);
    if (auto Err = Reader.read())
      return Err;
    Buf += FilenamesSize;

    // We'll read the coverage mapping records in the loop below.
    const char *CovBuf = Buf;
    Buf += CoverageSize;
    const char *CovEnd = Buf;

    if (Buf > End)
      return coveragemap_error::malformed;
    // Each coverage map has an alignment of 8, so we need to adjust alignment
    // before reading the next map.
    Buf += alignmentAdjustment(Buf, 8);

    while (FunBuf < FunEnd) {
      // Read the function information
      T NamePtr = endian::readNext<T, Endian, unaligned>(FunBuf);
      uint32_t NameSize = endian::readNext<uint32_t, Endian, unaligned>(FunBuf);
      uint32_t DataSize = endian::readNext<uint32_t, Endian, unaligned>(FunBuf);
      uint64_t FuncHash = endian::readNext<uint64_t, Endian, unaligned>(FunBuf);

      // Now use that to read the coverage data.
      if (CovBuf + DataSize > CovEnd)
        return coveragemap_error::malformed;
      auto Mapping = StringRef(CovBuf, DataSize);
      CovBuf += DataSize;

      // Ignore this record if we already have a record that points to the same
      // function name. This is useful to ignore the redundant records for the
      // functions with ODR linkage.
      if (!UniqueFunctionMappingData.insert(NamePtr).second)
        continue;

      // Finally, grab the name and create a record.
      StringRef FuncName;
      if (std::error_code EC = ProfileNames.get(NamePtr, NameSize, FuncName))
        return EC;
      Records.push_back(BinaryCoverageReader::ProfileMappingRecord(
          CoverageMappingVersion(Version), FuncName, FuncHash, Mapping,
          FilenamesBegin, Filenames.size() - FilenamesBegin));
    }
  }

  return std::error_code();
}

static const char *TestingFormatMagic = "llvmcovmtestdata";

static std::error_code loadTestingFormat(StringRef Data,
                                         SectionData &ProfileNames,
                                         StringRef &CoverageMapping,
                                         uint8_t &BytesInAddress,
                                         support::endianness &Endian) {
  BytesInAddress = 8;
  Endian = support::endianness::little;

  Data = Data.substr(StringRef(TestingFormatMagic).size());
  if (Data.size() < 1)
    return coveragemap_error::truncated;
  unsigned N = 0;
  auto ProfileNamesSize =
      decodeULEB128(reinterpret_cast<const uint8_t *>(Data.data()), &N);
  if (N > Data.size())
    return coveragemap_error::malformed;
  Data = Data.substr(N);
  if (Data.size() < 1)
    return coveragemap_error::truncated;
  N = 0;
  ProfileNames.Address =
      decodeULEB128(reinterpret_cast<const uint8_t *>(Data.data()), &N);
  if (N > Data.size())
    return coveragemap_error::malformed;
  Data = Data.substr(N);
  if (Data.size() < ProfileNamesSize)
    return coveragemap_error::malformed;
  ProfileNames.Data = Data.substr(0, ProfileNamesSize);
  CoverageMapping = Data.substr(ProfileNamesSize);
  return std::error_code();
}

#if 0 // HLSL Change Starts - remove support for object files
static ErrorOr<SectionRef> lookupSection(ObjectFile &OF, StringRef Name) {
  StringRef FoundName;
  for (const auto &Section : OF.sections()) {
    if (auto EC = Section.getName(FoundName))
      return EC;
    if (FoundName == Name)
      return Section;
  }
  return coveragemap_error::no_data_found;
}
#endif // HLSL Change Ends - remove support for object files

static std::error_code loadBinaryFormat(MemoryBufferRef ObjectBuffer,
                                        SectionData &ProfileNames,
                                        StringRef &CoverageMapping,
                                        uint8_t &BytesInAddress,
                                        support::endianness &Endian,
                                        StringRef Arch) {
#if 1 // HLSL Change Starts - remove support for object files
  return std::error_code();
#else
  auto BinOrErr = object::createBinary(ObjectBuffer);
  if (std::error_code EC = BinOrErr.getError())
    return EC;
  auto Bin = std::move(BinOrErr.get());
  std::unique_ptr<ObjectFile> OF;
  if (auto *Universal = dyn_cast<object::MachOUniversalBinary>(Bin.get())) {
    // If we have a universal binary, try to look up the object for the
    // appropriate architecture.
    auto ObjectFileOrErr = Universal->getObjectForArch(Arch);
    if (std::error_code EC = ObjectFileOrErr.getError())
      return EC;
    OF = std::move(ObjectFileOrErr.get());
  } else
  if (isa<object::ObjectFile>(Bin.get())) {
    // For any other object file, upcast and take ownership.
    OF.reset(cast<object::ObjectFile>(Bin.release()));
    // If we've asked for a particular arch, make sure they match.
    if (!Arch.empty() && OF->getArch() != Triple(Arch).getArch())
      return object_error::arch_not_found;
  } else
    // We can only handle object files.
    return coveragemap_error::malformed;

  // The coverage uses native pointer sizes for the object it's written in.
  BytesInAddress = OF->getBytesInAddress();
  Endian = OF->isLittleEndian() ? support::endianness::little
                                : support::endianness::big;

  // Look for the sections that we are interested in.
  auto NamesSection = lookupSection(*OF, "__llvm_prf_names");
  if (auto EC = NamesSection.getError())
    return EC;
  auto CoverageSection = lookupSection(*OF, "__llvm_covmap");
  if (auto EC = CoverageSection.getError())
    return EC;

  // Get the contents of the given sections.
  if (std::error_code EC = CoverageSection->getContents(CoverageMapping))
    return EC;
  if (std::error_code EC = ProfileNames.load(*NamesSection))
    return EC;

  return std::error_code();
#endif // HLSL Change Ends - remove support for object files
}

ErrorOr<std::unique_ptr<BinaryCoverageReader>>
BinaryCoverageReader::create(std::unique_ptr<MemoryBuffer> &ObjectBuffer,
                             StringRef Arch) {
  std::unique_ptr<BinaryCoverageReader> Reader(new BinaryCoverageReader());

  SectionData Profile;
  StringRef Coverage;
  uint8_t BytesInAddress;
  support::endianness Endian;
  std::error_code EC;
  if (ObjectBuffer->getBuffer().startswith(TestingFormatMagic))
    // This is a special format used for testing.
    EC = loadTestingFormat(ObjectBuffer->getBuffer(), Profile, Coverage,
                           BytesInAddress, Endian);
  else
    EC = loadBinaryFormat(ObjectBuffer->getMemBufferRef(), Profile, Coverage,
                          BytesInAddress, Endian, Arch);
  if (EC)
    return EC;

  if (BytesInAddress == 4 && Endian == support::endianness::little)
    EC = readCoverageMappingData<uint32_t, support::endianness::little>(
        Profile, Coverage, Reader->MappingRecords, Reader->Filenames);
  else if (BytesInAddress == 4 && Endian == support::endianness::big)
    EC = readCoverageMappingData<uint32_t, support::endianness::big>(
        Profile, Coverage, Reader->MappingRecords, Reader->Filenames);
  else if (BytesInAddress == 8 && Endian == support::endianness::little)
    EC = readCoverageMappingData<uint64_t, support::endianness::little>(
        Profile, Coverage, Reader->MappingRecords, Reader->Filenames);
  else if (BytesInAddress == 8 && Endian == support::endianness::big)
    EC = readCoverageMappingData<uint64_t, support::endianness::big>(
        Profile, Coverage, Reader->MappingRecords, Reader->Filenames);
  else
    return coveragemap_error::malformed;
  if (EC)
    return EC;
  return std::move(Reader);
}

std::error_code
BinaryCoverageReader::readNextRecord(CoverageMappingRecord &Record) {
  if (CurrentRecord >= MappingRecords.size())
    return coveragemap_error::eof;

  FunctionsFilenames.clear();
  Expressions.clear();
  MappingRegions.clear();
  auto &R = MappingRecords[CurrentRecord];
  RawCoverageMappingReader Reader(
      R.CoverageMapping,
      makeArrayRef(Filenames).slice(R.FilenamesBegin, R.FilenamesSize),
      FunctionsFilenames, Expressions, MappingRegions);
  if (auto Err = Reader.read())
    return Err;

  Record.FunctionName = R.FunctionName;
  Record.FunctionHash = R.FunctionHash;
  Record.Filenames = FunctionsFilenames;
  Record.Expressions = Expressions;
  Record.MappingRegions = MappingRegions;

  ++CurrentRecord;
  return std::error_code();
}
