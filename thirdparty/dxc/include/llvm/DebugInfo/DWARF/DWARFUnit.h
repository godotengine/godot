//===-- DWARFUnit.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_DWARFUNIT_H
#define LLVM_LIB_DEBUGINFO_DWARFUNIT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugAbbrev.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugInfoEntry.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugRangeList.h"
#include "llvm/DebugInfo/DWARF/DWARFRelocMap.h"
#include "llvm/DebugInfo/DWARF/DWARFSection.h"
#include <vector>

namespace llvm {

namespace object {
class ObjectFile;
}

class DWARFContext;
class DWARFDebugAbbrev;
class DWARFUnit;
class StringRef;
class raw_ostream;

/// Base class for all DWARFUnitSection classes. This provides the
/// functionality common to all unit types.
class DWARFUnitSectionBase {
public:
  /// Returns the Unit that contains the given section offset in the
  /// same section this Unit originated from.
  virtual DWARFUnit *getUnitForOffset(uint32_t Offset) const = 0;

  void parse(DWARFContext &C, const DWARFSection &Section);
  void parseDWO(DWARFContext &C, const DWARFSection &DWOSection);

protected:
  virtual void parseImpl(DWARFContext &Context, const DWARFSection &Section,
                         const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                         StringRef SOS, StringRef AOS, bool isLittleEndian) = 0;

  ~DWARFUnitSectionBase() = default;
};

/// Concrete instance of DWARFUnitSection, specialized for one Unit type.
template<typename UnitType>
class DWARFUnitSection final : public SmallVector<std::unique_ptr<UnitType>, 1>,
                               public DWARFUnitSectionBase {

  struct UnitOffsetComparator {
    bool operator()(uint32_t LHS,
                    const std::unique_ptr<UnitType> &RHS) const {
      return LHS < RHS->getNextUnitOffset();
    }
  };

  bool Parsed;

public:
  DWARFUnitSection() : Parsed(false) {}
  DWARFUnitSection(DWARFUnitSection &&DUS) :
    SmallVector<std::unique_ptr<UnitType>, 1>(std::move(DUS)), Parsed(DUS.Parsed) {}

  typedef llvm::SmallVectorImpl<std::unique_ptr<UnitType>> UnitVector;
  typedef typename UnitVector::iterator iterator;
  typedef llvm::iterator_range<typename UnitVector::iterator> iterator_range;

  UnitType *getUnitForOffset(uint32_t Offset) const override {
    auto *CU = std::upper_bound(this->begin(), this->end(), Offset,
                                UnitOffsetComparator());
    if (CU != this->end())
      return CU->get();
    return nullptr;
  }

private:
  void parseImpl(DWARFContext &Context, const DWARFSection &Section,
                 const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                 StringRef SOS, StringRef AOS, bool LE) override {
    if (Parsed)
      return;
    DataExtractor Data(Section.Data, LE, 0);
    uint32_t Offset = 0;
    while (Data.isValidOffset(Offset)) {
      auto U = llvm::make_unique<UnitType>(Context, Section, DA, RS, SS, SOS,
                                           AOS, LE, *this);
      if (!U->extract(Data, &Offset))
        break;
      this->push_back(std::move(U));
      Offset = this->back()->getNextUnitOffset();
    }
    Parsed = true;
  }
};

class DWARFUnit {
  DWARFContext &Context;
  // Section containing this DWARFUnit.
  const DWARFSection &InfoSection;

  const DWARFDebugAbbrev *Abbrev;
  StringRef RangeSection;
  uint32_t RangeSectionBase;
  StringRef StringSection;
  StringRef StringOffsetSection;
  StringRef AddrOffsetSection;
  uint32_t AddrOffsetSectionBase;
  bool isLittleEndian;
  const DWARFUnitSectionBase &UnitSection;

  uint32_t Offset;
  uint32_t Length;
  uint16_t Version;
  const DWARFAbbreviationDeclarationSet *Abbrevs;
  uint8_t AddrSize;
  uint64_t BaseAddr;
  // The compile unit debug information entry items.
  std::vector<DWARFDebugInfoEntryMinimal> DieArray;

  class DWOHolder {
    object::OwningBinary<object::ObjectFile> DWOFile;
    std::unique_ptr<DWARFContext> DWOContext;
    DWARFUnit *DWOU;
  public:
    DWOHolder(StringRef DWOPath);
    DWARFUnit *getUnit() const { return DWOU; }
  };
  std::unique_ptr<DWOHolder> DWO;

protected:
  virtual bool extractImpl(DataExtractor debug_info, uint32_t *offset_ptr);
  /// Size in bytes of the unit header.
  virtual uint32_t getHeaderSize() const { return 11; }

public:
  DWARFUnit(DWARFContext &Context, const DWARFSection &Section,
            const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
            StringRef SOS, StringRef AOS, bool LE,
            const DWARFUnitSectionBase &UnitSection);

  virtual ~DWARFUnit();

  DWARFContext& getContext() const { return Context; }

  StringRef getStringSection() const { return StringSection; }
  StringRef getStringOffsetSection() const { return StringOffsetSection; }
  void setAddrOffsetSection(StringRef AOS, uint32_t Base) {
    AddrOffsetSection = AOS;
    AddrOffsetSectionBase = Base;
  }
  void setRangesSection(StringRef RS, uint32_t Base) {
    RangeSection = RS;
    RangeSectionBase = Base;
  }

  bool getAddrOffsetSectionItem(uint32_t Index, uint64_t &Result) const;
  // FIXME: Result should be uint64_t in DWARF64.
  bool getStringOffsetSectionItem(uint32_t Index, uint32_t &Result) const;

  DataExtractor getDebugInfoExtractor() const {
    return DataExtractor(InfoSection.Data, isLittleEndian, AddrSize);
  }
  DataExtractor getStringExtractor() const {
    return DataExtractor(StringSection, false, 0);
  }

  const RelocAddrMap *getRelocMap() const { return &InfoSection.Relocs; }

  bool extract(DataExtractor debug_info, uint32_t* offset_ptr);

  /// extractRangeList - extracts the range list referenced by this compile
  /// unit from .debug_ranges section. Returns true on success.
  /// Requires that compile unit is already extracted.
  bool extractRangeList(uint32_t RangeListOffset,
                        DWARFDebugRangeList &RangeList) const;
  void clear();
  uint32_t getOffset() const { return Offset; }
  uint32_t getNextUnitOffset() const { return Offset + Length + 4; }
  uint32_t getLength() const { return Length; }
  uint16_t getVersion() const { return Version; }
  const DWARFAbbreviationDeclarationSet *getAbbreviations() const {
    return Abbrevs;
  }
  uint8_t getAddressByteSize() const { return AddrSize; }
  uint64_t getBaseAddress() const { return BaseAddr; }

  void setBaseAddress(uint64_t base_addr) {
    BaseAddr = base_addr;
  }

  const DWARFDebugInfoEntryMinimal *getUnitDIE(bool ExtractUnitDIEOnly = true) {
    extractDIEsIfNeeded(ExtractUnitDIEOnly);
    return DieArray.empty() ? nullptr : &DieArray[0];
  }

  const char *getCompilationDir();
  uint64_t getDWOId();

  void collectAddressRanges(DWARFAddressRangesVector &CURanges);

  /// getInlinedChainForAddress - fetches inlined chain for a given address.
  /// Returns empty chain if there is no subprogram containing address. The
  /// chain is valid as long as parsed compile unit DIEs are not cleared.
  DWARFDebugInfoEntryInlinedChain getInlinedChainForAddress(uint64_t Address);

  /// getUnitSection - Return the DWARFUnitSection containing this unit.
  const DWARFUnitSectionBase &getUnitSection() const { return UnitSection; }

  /// \brief Returns the number of DIEs in the unit. Parses the unit
  /// if necessary.
  unsigned getNumDIEs() {
    extractDIEsIfNeeded(false);
    return DieArray.size();
  }

  /// \brief Return the index of a DIE inside the unit's DIE vector.
  ///
  /// It is illegal to call this method with a DIE that hasn't be
  /// created by this unit. In other word, it's illegal to call this
  /// method on a DIE that isn't accessible by following
  /// children/sibling links starting from this unit's getUnitDIE().
  uint32_t getDIEIndex(const DWARFDebugInfoEntryMinimal *DIE) {
    assert(!DieArray.empty() && DIE >= &DieArray[0] &&
           DIE < &DieArray[0] + DieArray.size());
    return DIE - &DieArray[0];
  }

  /// \brief Return the DIE object at the given index.
  const DWARFDebugInfoEntryMinimal *getDIEAtIndex(unsigned Index) const {
    assert(Index < DieArray.size());
    return &DieArray[Index];
  }

  /// \brief Return the DIE object for a given offset inside the
  /// unit's DIE vector.
  ///
  /// The unit needs to have his DIEs extracted for this method to work.
  const DWARFDebugInfoEntryMinimal *getDIEForOffset(uint32_t Offset) const {
    assert(!DieArray.empty());
    auto it = std::lower_bound(
        DieArray.begin(), DieArray.end(), Offset,
        [=](const DWARFDebugInfoEntryMinimal &LHS, uint32_t Offset) {
          return LHS.getOffset() < Offset;
        });
    return it == DieArray.end() ? nullptr : &*it;
  }

private:
  /// Size in bytes of the .debug_info data associated with this compile unit.
  size_t getDebugInfoSize() const { return Length + 4 - getHeaderSize(); }

  /// extractDIEsIfNeeded - Parses a compile unit and indexes its DIEs if it
  /// hasn't already been done. Returns the number of DIEs parsed at this call.
  size_t extractDIEsIfNeeded(bool CUDieOnly);
  /// extractDIEsToVector - Appends all parsed DIEs to a vector.
  void extractDIEsToVector(bool AppendCUDie, bool AppendNonCUDIEs,
                           std::vector<DWARFDebugInfoEntryMinimal> &DIEs) const;
  /// setDIERelations - We read in all of the DIE entries into our flat list
  /// of DIE entries and now we need to go back through all of them and set the
  /// parent, sibling and child pointers for quick DIE navigation.
  void setDIERelations();
  /// clearDIEs - Clear parsed DIEs to keep memory usage low.
  void clearDIEs(bool KeepCUDie);

  /// parseDWO - Parses .dwo file for current compile unit. Returns true if
  /// it was actually constructed.
  bool parseDWO();

  /// getSubprogramForAddress - Returns subprogram DIE with address range
  /// encompassing the provided address. The pointer is alive as long as parsed
  /// compile unit DIEs are not cleared.
  const DWARFDebugInfoEntryMinimal *getSubprogramForAddress(uint64_t Address);
};

}

#endif
