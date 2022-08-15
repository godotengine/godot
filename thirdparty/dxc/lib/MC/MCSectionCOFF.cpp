//===- lib/MC/MCSectionCOFF.cpp - COFF Code Section Representation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSectionCOFF::~MCSectionCOFF() {} // anchor.

// ShouldOmitSectionDirective - Decides whether a '.section' directive
// should be printed before the section name
bool MCSectionCOFF::ShouldOmitSectionDirective(StringRef Name,
                                               const MCAsmInfo &MAI) const {
  if (COMDATSymbol)
    return false;

  // FIXME: Does .section .bss/.data/.text work everywhere??
  if (Name == ".text" || Name == ".data" || Name == ".bss")
    return true;

  return false;
}

void MCSectionCOFF::setSelection(int Selection) const {
  assert(Selection != 0 && "invalid COMDAT selection type");
  this->Selection = Selection;
  Characteristics |= COFF::IMAGE_SCN_LNK_COMDAT;
}

void MCSectionCOFF::PrintSwitchToSection(const MCAsmInfo &MAI,
                                         raw_ostream &OS,
                                         const MCExpr *Subsection) const {

  // standard sections don't require the '.section'
  if (ShouldOmitSectionDirective(SectionName, MAI)) {
    OS << '\t' << getSectionName() << '\n';
    return;
  }

  OS << "\t.section\t" << getSectionName() << ",\"";
  if (getCharacteristics() & COFF::IMAGE_SCN_CNT_INITIALIZED_DATA)
    OS << 'd';
  if (getCharacteristics() & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA)
    OS << 'b';
  if (getCharacteristics() & COFF::IMAGE_SCN_MEM_EXECUTE)
    OS << 'x';
  if (getCharacteristics() & COFF::IMAGE_SCN_MEM_WRITE)
    OS << 'w';
  else if (getCharacteristics() & COFF::IMAGE_SCN_MEM_READ)
    OS << 'r';
  else
    OS << 'y';
  if (getCharacteristics() & COFF::IMAGE_SCN_LNK_REMOVE)
    OS << 'n';
  if (getCharacteristics() & COFF::IMAGE_SCN_MEM_SHARED)
    OS << 's';
  OS << '"';

  if (getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT) {
    OS << ",";
    switch (Selection) {
      case COFF::IMAGE_COMDAT_SELECT_NODUPLICATES:
        OS << "one_only,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_ANY:
        OS << "discard,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_SAME_SIZE:
        OS << "same_size,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH:
        OS << "same_contents,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE:
        OS << "associative,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_LARGEST:
        OS << "largest,";
        break;
      case COFF::IMAGE_COMDAT_SELECT_NEWEST:
        OS << "newest,";
        break;
      default:
        assert (0 && "unsupported COFF selection type");
        break;
    }
    assert(COMDATSymbol);
    COMDATSymbol->print(OS, &MAI);
  }
  OS << '\n';
}

bool MCSectionCOFF::UseCodeAlign() const {
  return getKind().isText();
}

bool MCSectionCOFF::isVirtualSection() const {
  return getCharacteristics() & COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
}
