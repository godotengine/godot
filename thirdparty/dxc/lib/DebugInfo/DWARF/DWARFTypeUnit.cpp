//===-- DWARFTypeUnit.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

bool DWARFTypeUnit::extractImpl(DataExtractor debug_info,
                                uint32_t *offset_ptr) {
  if (!DWARFUnit::extractImpl(debug_info, offset_ptr))
    return false;
  TypeHash = debug_info.getU64(offset_ptr);
  TypeOffset = debug_info.getU32(offset_ptr);
  return TypeOffset < getLength();
}

void DWARFTypeUnit::dump(raw_ostream &OS) {
  OS << format("0x%08x", getOffset()) << ": Type Unit:"
     << " length = " << format("0x%08x", getLength())
     << " version = " << format("0x%04x", getVersion())
     << " abbr_offset = " << format("0x%04x", getAbbreviations()->getOffset())
     << " addr_size = " << format("0x%02x", getAddressByteSize())
     << " type_signature = " << format("0x%16" PRIx64, TypeHash)
     << " type_offset = " << format("0x%04x", TypeOffset)
     << " (next unit at " << format("0x%08x", getNextUnitOffset())
     << ")\n";

  if (const DWARFDebugInfoEntryMinimal *TU = getUnitDIE(false))
    TU->dump(OS, this, -1U);
  else
    OS << "<type unit can't be parsed!>\n\n";
}
