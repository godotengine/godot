//===- GCOV.h - LLVM coverage tool ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header provides the interface to read and write coverage files that
// use 'gcov' format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_GCOV_H
#define LLVM_SUPPORT_GCOV_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class GCOVFunction;
class GCOVBlock;
class FileInfo;

namespace GCOV {
enum GCOVVersion { V402, V404 };
} // end GCOV namespace

/// GCOVOptions - A struct for passing gcov options between functions.
struct GCOVOptions {
  GCOVOptions(bool A, bool B, bool C, bool F, bool P, bool U, bool L, bool N)
      : AllBlocks(A), BranchInfo(B), BranchCount(C), FuncCoverage(F),
        PreservePaths(P), UncondBranch(U), LongFileNames(L), NoOutput(N) {}

  bool AllBlocks;
  bool BranchInfo;
  bool BranchCount;
  bool FuncCoverage;
  bool PreservePaths;
  bool UncondBranch;
  bool LongFileNames;
  bool NoOutput;
};

/// GCOVBuffer - A wrapper around MemoryBuffer to provide GCOV specific
/// read operations.
class GCOVBuffer {
public:
  GCOVBuffer(MemoryBuffer *B) : Buffer(B), Cursor(0) {}

  /// readGCNOFormat - Check GCNO signature is valid at the beginning of buffer.
  bool readGCNOFormat() {
    StringRef File = Buffer->getBuffer().slice(0, 4);
    if (File != "oncg") {
      errs() << "Unexpected file type: " << File << ".\n";
      return false;
    }
    Cursor = 4;
    return true;
  }

  /// readGCDAFormat - Check GCDA signature is valid at the beginning of buffer.
  bool readGCDAFormat() {
    StringRef File = Buffer->getBuffer().slice(0, 4);
    if (File != "adcg") {
      errs() << "Unexpected file type: " << File << ".\n";
      return false;
    }
    Cursor = 4;
    return true;
  }

  /// readGCOVVersion - Read GCOV version.
  bool readGCOVVersion(GCOV::GCOVVersion &Version) {
    StringRef VersionStr = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (VersionStr == "*204") {
      Cursor += 4;
      Version = GCOV::V402;
      return true;
    }
    if (VersionStr == "*404") {
      Cursor += 4;
      Version = GCOV::V404;
      return true;
    }
    errs() << "Unexpected version: " << VersionStr << ".\n";
    return false;
  }

  /// readFunctionTag - If cursor points to a function tag then increment the
  /// cursor and return true otherwise return false.
  bool readFunctionTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\0' ||
        Tag[3] != '\1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readBlockTag - If cursor points to a block tag then increment the
  /// cursor and return true otherwise return false.
  bool readBlockTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\x41' ||
        Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readEdgeTag - If cursor points to an edge tag then increment the
  /// cursor and return true otherwise return false.
  bool readEdgeTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\x43' ||
        Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readLineTag - If cursor points to a line tag then increment the
  /// cursor and return true otherwise return false.
  bool readLineTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\x45' ||
        Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readArcTag - If cursor points to an gcda arc tag then increment the
  /// cursor and return true otherwise return false.
  bool readArcTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\xa1' ||
        Tag[3] != '\1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readObjectTag - If cursor points to an object summary tag then increment
  /// the cursor and return true otherwise return false.
  bool readObjectTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\0' ||
        Tag[3] != '\xa1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readProgramTag - If cursor points to a program summary tag then increment
  /// the cursor and return true otherwise return false.
  bool readProgramTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    if (Tag.empty() || Tag[0] != '\0' || Tag[1] != '\0' || Tag[2] != '\0' ||
        Tag[3] != '\xa3') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  bool readInt(uint32_t &Val) {
    if (Buffer->getBuffer().size() < Cursor + 4) {
      errs() << "Unexpected end of memory buffer: " << Cursor + 4 << ".\n";
      return false;
    }
    StringRef Str = Buffer->getBuffer().slice(Cursor, Cursor + 4);
    Cursor += 4;
    Val = *(const uint32_t *)(Str.data());
    return true;
  }

  bool readInt64(uint64_t &Val) {
    uint32_t Lo, Hi;
    if (!readInt(Lo) || !readInt(Hi))
      return false;
    Val = ((uint64_t)Hi << 32) | Lo;
    return true;
  }

  bool readString(StringRef &Str) {
    uint32_t Len = 0;
    // Keep reading until we find a non-zero length. This emulates gcov's
    // behaviour, which appears to do the same.
    while (Len == 0)
      if (!readInt(Len))
        return false;
    Len *= 4;
    if (Buffer->getBuffer().size() < Cursor + Len) {
      errs() << "Unexpected end of memory buffer: " << Cursor + Len << ".\n";
      return false;
    }
    Str = Buffer->getBuffer().slice(Cursor, Cursor + Len).split('\0').first;
    Cursor += Len;
    return true;
  }

  uint64_t getCursor() const { return Cursor; }
  void advanceCursor(uint32_t n) { Cursor += n * 4; }

private:
  MemoryBuffer *Buffer;
  uint64_t Cursor;
};

/// GCOVFile - Collects coverage information for one pair of coverage file
/// (.gcno and .gcda).
class GCOVFile {
public:
  GCOVFile()
      : GCNOInitialized(false), Checksum(0), Functions(), RunCount(0),
        ProgramCount(0) {}
  bool readGCNO(GCOVBuffer &Buffer);
  bool readGCDA(GCOVBuffer &Buffer);
  uint32_t getChecksum() const { return Checksum; }
  void dump() const;
  void collectLineCounts(FileInfo &FI);

private:
  bool GCNOInitialized;
  GCOV::GCOVVersion Version;
  uint32_t Checksum;
  SmallVector<std::unique_ptr<GCOVFunction>, 16> Functions;
  uint32_t RunCount;
  uint32_t ProgramCount;
};

/// GCOVEdge - Collects edge information.
struct GCOVEdge {
  GCOVEdge(GCOVBlock &S, GCOVBlock &D) : Src(S), Dst(D), Count(0) {}

  GCOVBlock &Src;
  GCOVBlock &Dst;
  uint64_t Count;
};

/// GCOVFunction - Collects function information.
class GCOVFunction {
public:
  typedef pointee_iterator<SmallVectorImpl<
      std::unique_ptr<GCOVBlock>>::const_iterator> BlockIterator;

  GCOVFunction(GCOVFile &P) : Parent(P), Ident(0), LineNumber(0) {}
  bool readGCNO(GCOVBuffer &Buffer, GCOV::GCOVVersion Version);
  bool readGCDA(GCOVBuffer &Buffer, GCOV::GCOVVersion Version);
  StringRef getName() const { return Name; }
  StringRef getFilename() const { return Filename; }
  size_t getNumBlocks() const { return Blocks.size(); }
  uint64_t getEntryCount() const;
  uint64_t getExitCount() const;

  BlockIterator block_begin() const { return Blocks.begin(); }
  BlockIterator block_end() const { return Blocks.end(); }
  iterator_range<BlockIterator> blocks() const {
    return make_range(block_begin(), block_end());
  }

  void dump() const;
  void collectLineCounts(FileInfo &FI);

private:
  GCOVFile &Parent;
  uint32_t Ident;
  uint32_t Checksum;
  uint32_t LineNumber;
  StringRef Name;
  StringRef Filename;
  SmallVector<std::unique_ptr<GCOVBlock>, 16> Blocks;
  SmallVector<std::unique_ptr<GCOVEdge>, 16> Edges;
};

/// GCOVBlock - Collects block information.
class GCOVBlock {
  struct EdgeWeight {
    EdgeWeight(GCOVBlock *D) : Dst(D), Count(0) {}

    GCOVBlock *Dst;
    uint64_t Count;
  };

  struct SortDstEdgesFunctor {
    bool operator()(const GCOVEdge *E1, const GCOVEdge *E2) {
      return E1->Dst.Number < E2->Dst.Number;
    }
  };

public:
  typedef SmallVectorImpl<GCOVEdge *>::const_iterator EdgeIterator;

  GCOVBlock(GCOVFunction &P, uint32_t N)
      : Parent(P), Number(N), Counter(0), DstEdgesAreSorted(true), SrcEdges(),
        DstEdges(), Lines() {}
  ~GCOVBlock();
  const GCOVFunction &getParent() const { return Parent; }
  void addLine(uint32_t N) { Lines.push_back(N); }
  uint32_t getLastLine() const { return Lines.back(); }
  void addCount(size_t DstEdgeNo, uint64_t N);
  uint64_t getCount() const { return Counter; }

  void addSrcEdge(GCOVEdge *Edge) {
    assert(&Edge->Dst == this); // up to caller to ensure edge is valid
    SrcEdges.push_back(Edge);
  }
  void addDstEdge(GCOVEdge *Edge) {
    assert(&Edge->Src == this); // up to caller to ensure edge is valid
    // Check if adding this edge causes list to become unsorted.
    if (DstEdges.size() && DstEdges.back()->Dst.Number > Edge->Dst.Number)
      DstEdgesAreSorted = false;
    DstEdges.push_back(Edge);
  }
  size_t getNumSrcEdges() const { return SrcEdges.size(); }
  size_t getNumDstEdges() const { return DstEdges.size(); }
  void sortDstEdges();

  EdgeIterator src_begin() const { return SrcEdges.begin(); }
  EdgeIterator src_end() const { return SrcEdges.end(); }
  iterator_range<EdgeIterator> srcs() const {
    return make_range(src_begin(), src_end());
  }

  EdgeIterator dst_begin() const { return DstEdges.begin(); }
  EdgeIterator dst_end() const { return DstEdges.end(); }
  iterator_range<EdgeIterator> dsts() const {
    return make_range(dst_begin(), dst_end());
  }

  void dump() const;
  void collectLineCounts(FileInfo &FI);

private:
  GCOVFunction &Parent;
  uint32_t Number;
  uint64_t Counter;
  bool DstEdgesAreSorted;
  SmallVector<GCOVEdge *, 16> SrcEdges;
  SmallVector<GCOVEdge *, 16> DstEdges;
  SmallVector<uint32_t, 16> Lines;
};

class FileInfo {
  // It is unlikely--but possible--for multiple functions to be on the same
  // line.
  // Therefore this typedef allows LineData.Functions to store multiple
  // functions
  // per instance. This is rare, however, so optimize for the common case.
  typedef SmallVector<const GCOVFunction *, 1> FunctionVector;
  typedef DenseMap<uint32_t, FunctionVector> FunctionLines;
  typedef SmallVector<const GCOVBlock *, 4> BlockVector;
  typedef DenseMap<uint32_t, BlockVector> BlockLines;

  struct LineData {
    LineData() : LastLine(0) {}
    BlockLines Blocks;
    FunctionLines Functions;
    uint32_t LastLine;
  };

  struct GCOVCoverage {
    GCOVCoverage(StringRef Name)
        : Name(Name), LogicalLines(0), LinesExec(0), Branches(0),
          BranchesExec(0), BranchesTaken(0) {}

    StringRef Name;

    uint32_t LogicalLines;
    uint32_t LinesExec;

    uint32_t Branches;
    uint32_t BranchesExec;
    uint32_t BranchesTaken;
  };

public:
  FileInfo(const GCOVOptions &Options)
      : Options(Options), LineInfo(), RunCount(0), ProgramCount(0) {}

  void addBlockLine(StringRef Filename, uint32_t Line, const GCOVBlock *Block) {
    if (Line > LineInfo[Filename].LastLine)
      LineInfo[Filename].LastLine = Line;
    LineInfo[Filename].Blocks[Line - 1].push_back(Block);
  }
  void addFunctionLine(StringRef Filename, uint32_t Line,
                       const GCOVFunction *Function) {
    if (Line > LineInfo[Filename].LastLine)
      LineInfo[Filename].LastLine = Line;
    LineInfo[Filename].Functions[Line - 1].push_back(Function);
  }
  void setRunCount(uint32_t Runs) { RunCount = Runs; }
  void setProgramCount(uint32_t Programs) { ProgramCount = Programs; }
  void print(raw_ostream &OS, StringRef MainFilename, StringRef GCNOFile,
             StringRef GCDAFile);

private:
  std::string getCoveragePath(StringRef Filename, StringRef MainFilename);
  std::unique_ptr<raw_ostream> openCoveragePath(StringRef CoveragePath);
  void printFunctionSummary(raw_ostream &OS, const FunctionVector &Funcs) const;
  void printBlockInfo(raw_ostream &OS, const GCOVBlock &Block,
                      uint32_t LineIndex, uint32_t &BlockNo) const;
  void printBranchInfo(raw_ostream &OS, const GCOVBlock &Block,
                       GCOVCoverage &Coverage, uint32_t &EdgeNo);
  void printUncondBranchInfo(raw_ostream &OS, uint32_t &EdgeNo,
                             uint64_t Count) const;

  void printCoverage(raw_ostream &OS, const GCOVCoverage &Coverage) const;
  void printFuncCoverage(raw_ostream &OS) const;
  void printFileCoverage(raw_ostream &OS) const;

  const GCOVOptions &Options;
  StringMap<LineData> LineInfo;
  uint32_t RunCount;
  uint32_t ProgramCount;

  typedef SmallVector<std::pair<std::string, GCOVCoverage>, 4> FileCoverageList;
  typedef MapVector<const GCOVFunction *, GCOVCoverage> FuncCoverageMap;

  FileCoverageList FileCoverages;
  FuncCoverageMap FuncCoverages;
};
}

#endif
