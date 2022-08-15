//===-- DWARFDebugRangesList.cpp ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void DWARFDebugRangeList::clear() {
  Offset = -1U;
  AddressSize = 0;
  Entries.clear();
}

bool DWARFDebugRangeList::extract(DataExtractor data, uint32_t *offset_ptr) {
  clear();
  if (!data.isValidOffset(*offset_ptr))
    return false;
  AddressSize = data.getAddressSize();
  if (AddressSize != 4 && AddressSize != 8)
    return false;
  Offset = *offset_ptr;
  while (true) {
    RangeListEntry entry;
    uint32_t prev_offset = *offset_ptr;
    entry.StartAddress = data.getAddress(offset_ptr);
    entry.EndAddress = data.getAddress(offset_ptr);
    // Check that both values were extracted correctly.
    if (*offset_ptr != prev_offset + 2 * AddressSize) {
      clear();
      return false;
    }
    if (entry.isEndOfListEntry())
      break;
    Entries.push_back(entry);
  }
  return true;
}

void DWARFDebugRangeList::dump(raw_ostream &OS) const {
  for (const RangeListEntry &RLE : Entries) {
    const char *format_str = (AddressSize == 4
                              ? "%08x %08"  PRIx64 " %08"  PRIx64 "\n"
                              : "%08x %016" PRIx64 " %016" PRIx64 "\n");
    OS << format(format_str, Offset, RLE.StartAddress, RLE.EndAddress);
  }
  OS << format("%08x <End of list>\n", Offset);
}

DWARFAddressRangesVector
DWARFDebugRangeList::getAbsoluteRanges(uint64_t BaseAddress) const {
  DWARFAddressRangesVector Res;
  for (const RangeListEntry &RLE : Entries) {
    if (RLE.isBaseAddressSelectionEntry(AddressSize)) {
      BaseAddress = RLE.EndAddress;
    } else {
      Res.push_back(std::make_pair(BaseAddress + RLE.StartAddress,
                                   BaseAddress + RLE.EndAddress));
    }
  }
  return Res;
}
