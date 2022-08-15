//===-- DWARFDebugAbbrev.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFDEBUGABBREV_H
#define LLVM_LIB_DEBUGINFO_DWARFDEBUGABBREV_H

#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include <list>
#include <map>
#include <vector>

namespace llvm {

class DWARFAbbreviationDeclarationSet {
  uint32_t Offset;
  /// Code of the first abbreviation, if all abbreviations in the set have
  /// consecutive codes. UINT32_MAX otherwise.
  uint32_t FirstAbbrCode;
  std::vector<DWARFAbbreviationDeclaration> Decls;

public:
  DWARFAbbreviationDeclarationSet();

  uint32_t getOffset() const { return Offset; }
  void dump(raw_ostream &OS) const;
  bool extract(DataExtractor Data, uint32_t *OffsetPtr);

  const DWARFAbbreviationDeclaration *
  getAbbreviationDeclaration(uint32_t AbbrCode) const;

private:
  void clear();
};

class DWARFDebugAbbrev {
  typedef std::map<uint64_t, DWARFAbbreviationDeclarationSet>
    DWARFAbbreviationDeclarationSetMap;

  DWARFAbbreviationDeclarationSetMap AbbrDeclSets;
  mutable DWARFAbbreviationDeclarationSetMap::const_iterator PrevAbbrOffsetPos;

public:
  DWARFDebugAbbrev();

  const DWARFAbbreviationDeclarationSet *
  getAbbreviationDeclarationSet(uint64_t CUAbbrOffset) const;

  void dump(raw_ostream &OS) const;
  void extract(DataExtractor Data);

private:
  void clear();
};

}

#endif
