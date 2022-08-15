//===-- DWARFDebugAbbrev.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

DWARFAbbreviationDeclarationSet::DWARFAbbreviationDeclarationSet() {
  clear();
}

void DWARFAbbreviationDeclarationSet::clear() {
  Offset = 0;
  FirstAbbrCode = 0;
  Decls.clear();
}

bool DWARFAbbreviationDeclarationSet::extract(DataExtractor Data,
                                              uint32_t *OffsetPtr) {
  clear();
  const uint32_t BeginOffset = *OffsetPtr;
  Offset = BeginOffset;
  DWARFAbbreviationDeclaration AbbrDecl;
  uint32_t PrevAbbrCode = 0;
  while (AbbrDecl.extract(Data, OffsetPtr)) {
    if (FirstAbbrCode == 0) {
      FirstAbbrCode = AbbrDecl.getCode();
    } else {
      if (PrevAbbrCode + 1 != AbbrDecl.getCode()) {
        // Codes are not consecutive, can't do O(1) lookups.
        FirstAbbrCode = UINT32_MAX;
      }
    }
    PrevAbbrCode = AbbrDecl.getCode();
    Decls.push_back(std::move(AbbrDecl));
  }
  return BeginOffset != *OffsetPtr;
}

void DWARFAbbreviationDeclarationSet::dump(raw_ostream &OS) const {
  for (const auto &Decl : Decls)
    Decl.dump(OS);
}

const DWARFAbbreviationDeclaration *
DWARFAbbreviationDeclarationSet::getAbbreviationDeclaration(
    uint32_t AbbrCode) const {
  if (FirstAbbrCode == UINT32_MAX) {
    for (const auto &Decl : Decls) {
      if (Decl.getCode() == AbbrCode)
        return &Decl;
    }
    return nullptr;
  }
  if (AbbrCode < FirstAbbrCode || AbbrCode >= FirstAbbrCode + Decls.size())
    return nullptr;
  return &Decls[AbbrCode - FirstAbbrCode];
}

DWARFDebugAbbrev::DWARFDebugAbbrev() {
  clear();
}

void DWARFDebugAbbrev::clear() {
  AbbrDeclSets.clear();
  PrevAbbrOffsetPos = AbbrDeclSets.end();
}

void DWARFDebugAbbrev::extract(DataExtractor Data) {
  clear();

  uint32_t Offset = 0;
  DWARFAbbreviationDeclarationSet AbbrDecls;
  while (Data.isValidOffset(Offset)) {
    uint32_t CUAbbrOffset = Offset;
    if (!AbbrDecls.extract(Data, &Offset))
      break;
    AbbrDeclSets[CUAbbrOffset] = std::move(AbbrDecls);
  }
}

void DWARFDebugAbbrev::dump(raw_ostream &OS) const {
  if (AbbrDeclSets.empty()) {
    OS << "< EMPTY >\n";
    return;
  }

  for (const auto &I : AbbrDeclSets) {
    OS << format("Abbrev table for offset: 0x%8.8" PRIx64 "\n", I.first);
    I.second.dump(OS);
  }
}

const DWARFAbbreviationDeclarationSet*
DWARFDebugAbbrev::getAbbreviationDeclarationSet(uint64_t CUAbbrOffset) const {
  const auto End = AbbrDeclSets.end();
  if (PrevAbbrOffsetPos != End && PrevAbbrOffsetPos->first == CUAbbrOffset) {
    return &(PrevAbbrOffsetPos->second);
  }

  const auto Pos = AbbrDeclSets.find(CUAbbrOffset);
  if (Pos != End) {
    PrevAbbrOffsetPos = Pos;
    return &(Pos->second);
  }

  return nullptr;
}
