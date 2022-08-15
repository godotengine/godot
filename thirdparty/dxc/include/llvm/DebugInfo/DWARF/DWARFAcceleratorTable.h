//===--- DWARFAcceleratorTable.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFACCELERATORTABLE_H
#define LLVM_LIB_DEBUGINFO_DWARFACCELERATORTABLE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include <cstdint>

namespace llvm {

class DWARFAcceleratorTable {

  struct Header {
    uint32_t Magic;
    uint16_t Version;
    uint16_t HashFunction;
    uint32_t NumBuckets;
    uint32_t NumHashes;
    uint32_t HeaderDataLength;
  };

  struct HeaderData {
    typedef uint16_t AtomType;
    typedef uint16_t Form;
    uint32_t DIEOffsetBase;
    SmallVector<std::pair<AtomType, Form>, 3> Atoms;
  };

  struct Header Hdr;
  struct HeaderData HdrData;
  DataExtractor AccelSection;
  DataExtractor StringSection;
  const RelocAddrMap& Relocs;
public:
  DWARFAcceleratorTable(DataExtractor AccelSection, DataExtractor StringSection,
                        const RelocAddrMap &Relocs)
    : AccelSection(AccelSection), StringSection(StringSection), Relocs(Relocs) {}

  bool extract();
  void dump(raw_ostream &OS) const;
};

}

#endif
