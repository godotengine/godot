//===-- DWARFDebugLoc.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFDEBUGLOC_H
#define LLVM_LIB_DEBUGINFO_DWARFDEBUGLOC_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

class raw_ostream;

class DWARFDebugLoc {
  /// A single location within a location list.
  struct Entry {
    /// The beginning address of the instruction range.
    uint64_t Begin;
    /// The ending address of the instruction range.
    uint64_t End;
    /// The location of the variable within the specified range.
    SmallVector<unsigned char, 4> Loc;
  };

  /// A list of locations that contain one variable.
  struct LocationList {
    /// The beginning offset where this location list is stored in the debug_loc
    /// section.
    unsigned Offset;
    /// All the locations in which the variable is stored.
    SmallVector<Entry, 2> Entries;
  };

  typedef SmallVector<LocationList, 4> LocationLists;

  /// A list of all the variables in the debug_loc section, each one describing
  /// the locations in which the variable is stored.
  LocationLists Locations;

  /// A map used to resolve binary relocations.
  const RelocAddrMap &RelocMap;

public:
  DWARFDebugLoc(const RelocAddrMap &LocRelocMap) : RelocMap(LocRelocMap) {}
  /// Print the location lists found within the debug_loc section.
  void dump(raw_ostream &OS) const;
  /// Parse the debug_loc section accessible via the 'data' parameter using the
  /// specified address size to interpret the address ranges.
  void parse(DataExtractor data, unsigned AddressSize);
};

class DWARFDebugLocDWO {
  struct Entry {
    uint64_t Start;
    uint32_t Length;
    SmallVector<unsigned char, 4> Loc;
  };

  struct LocationList {
    unsigned Offset;
    SmallVector<Entry, 2> Entries;
  };

  typedef SmallVector<LocationList, 4> LocationLists;

  LocationLists Locations;

public:
  void parse(DataExtractor data);
  void dump(raw_ostream &OS) const;
};
}

#endif
