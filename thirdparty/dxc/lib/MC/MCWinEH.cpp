//===- lib/MC/MCWinEH.cpp - Windows EH implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCWinEH.h"
#include "llvm/Support/COFF.h"

namespace llvm {
namespace WinEH {

/// We can't have one section for all .pdata or .xdata because the Microsoft
/// linker seems to want all code relocations to refer to the same object file
/// section. If the code described is comdat, create a new comdat section
/// associated with that comdat. If the code described is not in the main .text
/// section, make a new section for it. Otherwise use the main unwind info
/// section.
static MCSection *getUnwindInfoSection(StringRef SecName,
                                       MCSectionCOFF *UnwindSec,
                                       const MCSymbol *Function,
                                       MCContext &Context) {
  if (Function && Function->isInSection()) {
    // If Function is in a COMDAT, get or create an unwind info section in that
    // COMDAT group.
    const MCSectionCOFF *FunctionSection =
        cast<MCSectionCOFF>(&Function->getSection());
    if (FunctionSection->getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT) {
      return Context.getAssociativeCOFFSection(
          UnwindSec, FunctionSection->getCOMDATSymbol());
    }

    // If Function is in a section other than .text, create a new .pdata section.
    // Otherwise use the plain .pdata section.
    if (const auto *Section = dyn_cast<MCSectionCOFF>(FunctionSection)) {
      StringRef CodeSecName = Section->getSectionName();
      if (CodeSecName == ".text")
        return UnwindSec;

      if (CodeSecName.startswith(".text$"))
        CodeSecName = CodeSecName.substr(6);

      return Context.getCOFFSection(
          (SecName + Twine('$') + CodeSecName).str(),
          COFF::IMAGE_SCN_CNT_INITIALIZED_DATA | COFF::IMAGE_SCN_MEM_READ,
          SectionKind::getDataRel());
    }
  }

  return UnwindSec;

}

MCSection *UnwindEmitter::getPDataSection(const MCSymbol *Function,
                                          MCContext &Context) {
  MCSectionCOFF *PData =
      cast<MCSectionCOFF>(Context.getObjectFileInfo()->getPDataSection());
  return getUnwindInfoSection(".pdata", PData, Function, Context);
}

MCSection *UnwindEmitter::getXDataSection(const MCSymbol *Function,
                                          MCContext &Context) {
  MCSectionCOFF *XData =
      cast<MCSectionCOFF>(Context.getObjectFileInfo()->getXDataSection());
  return getUnwindInfoSection(".xdata", XData, Function, Context);
}

}
}

