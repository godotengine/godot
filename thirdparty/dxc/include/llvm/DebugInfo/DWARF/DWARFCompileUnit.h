//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFCOMPILEUNIT_H
#define LLVM_LIB_DEBUGINFO_DWARFCOMPILEUNIT_H

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"

namespace llvm {

class DWARFCompileUnit : public DWARFUnit {
public:
  DWARFCompileUnit(DWARFContext &Context, const DWARFSection &Section,
                   const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                   StringRef SOS, StringRef AOS, bool LE,
                   const DWARFUnitSectionBase &UnitSection)
      : DWARFUnit(Context, Section, DA, RS, SS, SOS, AOS, LE, UnitSection) {}
  void dump(raw_ostream &OS);
  // VTable anchor.
  ~DWARFCompileUnit() override;
};

}

#endif
