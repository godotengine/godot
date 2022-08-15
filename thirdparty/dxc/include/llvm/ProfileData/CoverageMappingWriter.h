//=-- CoverageMappingWriter.h - Code coverage mapping writer ------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing coverage mapping data for
// instrumentation based coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_COVERAGEMAPPINGWRITER_H
#define LLVM_PROFILEDATA_COVERAGEMAPPINGWRITER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace coverage {

/// \brief Writer of the filenames section for the instrumentation
/// based code coverage.
class CoverageFilenamesSectionWriter {
  ArrayRef<StringRef> Filenames;

public:
  CoverageFilenamesSectionWriter(ArrayRef<StringRef> Filenames)
      : Filenames(Filenames) {}

  /// \brief Write encoded filenames to the given output stream.
  void write(raw_ostream &OS);
};

/// \brief Writer for instrumentation based coverage mapping data.
class CoverageMappingWriter {
  ArrayRef<unsigned> VirtualFileMapping;
  ArrayRef<CounterExpression> Expressions;
  MutableArrayRef<CounterMappingRegion> MappingRegions;

public:
  CoverageMappingWriter(ArrayRef<unsigned> VirtualFileMapping,
                        ArrayRef<CounterExpression> Expressions,
                        MutableArrayRef<CounterMappingRegion> MappingRegions)
      : VirtualFileMapping(VirtualFileMapping), Expressions(Expressions),
        MappingRegions(MappingRegions) {}

  CoverageMappingWriter(ArrayRef<CounterExpression> Expressions,
                        MutableArrayRef<CounterMappingRegion> MappingRegions)
      : Expressions(Expressions), MappingRegions(MappingRegions) {}

  /// \brief Write encoded coverage mapping data to the given output stream.
  void write(raw_ostream &OS);
};

} // end namespace coverage
} // end namespace llvm

#endif
