//===-- llvm/CodeGen/DwarfFile.cpp - Dwarf Debug Framework ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DwarfFile.h"
#include "DwarfDebug.h"
#include "DwarfUnit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
DwarfFile::DwarfFile(AsmPrinter *AP, StringRef Pref, BumpPtrAllocator &DA)
    : Asm(AP), StrPool(DA, *Asm, Pref) {}

DwarfFile::~DwarfFile() {
  for (DIEAbbrev *Abbrev : Abbreviations)
    Abbrev->~DIEAbbrev();
}

// Define a unique number for the abbreviation.
//
DIEAbbrev &DwarfFile::assignAbbrevNumber(DIE &Die) {
  FoldingSetNodeID ID;
  DIEAbbrev Abbrev = Die.generateAbbrev();
  Abbrev.Profile(ID);

  void *InsertPos;
  if (DIEAbbrev *Existing =
          AbbreviationsSet.FindNodeOrInsertPos(ID, InsertPos)) {
    Die.setAbbrevNumber(Existing->getNumber());
    return *Existing;
  }

  // Move the abbreviation to the heap and assign a number.
  DIEAbbrev *New = new (AbbrevAllocator) DIEAbbrev(std::move(Abbrev));
  Abbreviations.push_back(New);
  New->setNumber(Abbreviations.size());
  Die.setAbbrevNumber(Abbreviations.size());

  // Store it for lookup.
  AbbreviationsSet.InsertNode(New, InsertPos);
  return *New;
}

void DwarfFile::addUnit(std::unique_ptr<DwarfUnit> U) {
  CUs.push_back(std::move(U));
}

// Emit the various dwarf units to the unit section USection with
// the abbreviations going into ASection.
void DwarfFile::emitUnits(bool UseOffsets) {
  for (const auto &TheU : CUs) {
    DIE &Die = TheU->getUnitDie();
    MCSection *USection = TheU->getSection();
    Asm->OutStreamer->SwitchSection(USection);

    TheU->emitHeader(UseOffsets);

    Asm->emitDwarfDIE(Die);
  }
}

// Compute the size and offset for each DIE.
void DwarfFile::computeSizeAndOffsets() {
  // Offset from the first CU in the debug info section is 0 initially.
  unsigned SecOffset = 0;

  // Iterate over each compile unit and set the size and offsets for each
  // DIE within each compile unit. All offsets are CU relative.
  for (const auto &TheU : CUs) {
    TheU->setDebugInfoOffset(SecOffset);

    // CU-relative offset is reset to 0 here.
    unsigned Offset = sizeof(int32_t) +      // Length of Unit Info
                      TheU->getHeaderSize(); // Unit-specific headers

    // EndOffset here is CU-relative, after laying out
    // all of the CU DIE.
    unsigned EndOffset = computeSizeAndOffset(TheU->getUnitDie(), Offset);
    SecOffset += EndOffset;
  }
}
// Compute the size and offset of a DIE. The offset is relative to start of the
// CU. It returns the offset after laying out the DIE.
unsigned DwarfFile::computeSizeAndOffset(DIE &Die, unsigned Offset) {
  // Record the abbreviation.
  const DIEAbbrev &Abbrev = assignAbbrevNumber(Die);

  // Set DIE offset
  Die.setOffset(Offset);

  // Start the size with the size of abbreviation code.
  Offset += getULEB128Size(Die.getAbbrevNumber());

  // Size the DIE attribute values.
  for (const auto &V : Die.values())
    // Size attribute value.
    Offset += V.SizeOf(Asm);

  // Size the DIE children if any.
  if (Die.hasChildren()) {
    (void)Abbrev;
    assert(Abbrev.hasChildren() && "Children flag not set");

    for (auto &Child : Die.children())
      Offset = computeSizeAndOffset(Child, Offset);

    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die.setSize(Offset - Die.getOffset());
  return Offset;
}

void DwarfFile::emitAbbrevs(MCSection *Section) {
  // Check to see if it is worth the effort.
  if (!Abbreviations.empty()) {
    // Start the debug abbrev section.
    Asm->OutStreamer->SwitchSection(Section);
    Asm->emitDwarfAbbrevs(Abbreviations);
  }
}

// Emit strings into a string section.
void DwarfFile::emitStrings(MCSection *StrSection, MCSection *OffsetSection) {
  StrPool.emit(*Asm, StrSection, OffsetSection);
}

bool DwarfFile::addScopeVariable(LexicalScope *LS, DbgVariable *Var) {
  SmallVectorImpl<DbgVariable *> &Vars = ScopeVariables[LS];
  const DILocalVariable *DV = Var->getVariable();
  // Variables with positive arg numbers are parameters.
  if (unsigned ArgNum = DV->getArg()) {
    // Keep all parameters in order at the start of the variable list to ensure
    // function types are correct (no out-of-order parameters)
    //
    // This could be improved by only doing it for optimized builds (unoptimized
    // builds have the right order to begin with), searching from the back (this
    // would catch the unoptimized case quickly), or doing a binary search
    // rather than linear search.
    auto I = Vars.begin();
    while (I != Vars.end()) {
      unsigned CurNum = (*I)->getVariable()->getArg();
      // A local (non-parameter) variable has been found, insert immediately
      // before it.
      if (CurNum == 0)
        break;
      // A later indexed parameter has been found, insert immediately before it.
      if (CurNum > ArgNum)
        break;
      if (CurNum == ArgNum) {
        (*I)->addMMIEntry(*Var);
        return false;
      }
      ++I;
    }
    Vars.insert(I, Var);
    return true;
  }

  Vars.push_back(Var);
  return true;
}
}
