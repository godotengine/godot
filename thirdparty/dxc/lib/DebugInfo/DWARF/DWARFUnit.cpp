//===-- DWARFUnit.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Path.h"
#include <cstdio>

using namespace llvm;
using namespace dwarf;

void DWARFUnitSectionBase::parse(DWARFContext &C, const DWARFSection &Section) {
  parseImpl(C, Section, C.getDebugAbbrev(), C.getRangeSection(),
            C.getStringSection(), StringRef(), C.getAddrSection(),
            C.isLittleEndian());
}

void DWARFUnitSectionBase::parseDWO(DWARFContext &C,
                                    const DWARFSection &DWOSection) {
  parseImpl(C, DWOSection, C.getDebugAbbrevDWO(), C.getRangeDWOSection(),
            C.getStringDWOSection(), C.getStringOffsetDWOSection(),
            C.getAddrSection(), C.isLittleEndian());
}

DWARFUnit::DWARFUnit(DWARFContext &DC, const DWARFSection &Section,
                     const DWARFDebugAbbrev *DA, StringRef RS, StringRef SS,
                     StringRef SOS, StringRef AOS, bool LE,
                     const DWARFUnitSectionBase &UnitSection)
    : Context(DC), InfoSection(Section), Abbrev(DA), RangeSection(RS),
      StringSection(SS), StringOffsetSection(SOS), AddrOffsetSection(AOS),
      isLittleEndian(LE), UnitSection(UnitSection) {
  clear();
}

DWARFUnit::~DWARFUnit() {
}

bool DWARFUnit::getAddrOffsetSectionItem(uint32_t Index,
                                                uint64_t &Result) const {
  uint32_t Offset = AddrOffsetSectionBase + Index * AddrSize;
  if (AddrOffsetSection.size() < Offset + AddrSize)
    return false;
  DataExtractor DA(AddrOffsetSection, isLittleEndian, AddrSize);
  Result = DA.getAddress(&Offset);
  return true;
}

bool DWARFUnit::getStringOffsetSectionItem(uint32_t Index,
                                                  uint32_t &Result) const {
  // FIXME: string offset section entries are 8-byte for DWARF64.
  const uint32_t ItemSize = 4;
  uint32_t Offset = Index * ItemSize;
  if (StringOffsetSection.size() < Offset + ItemSize)
    return false;
  DataExtractor DA(StringOffsetSection, isLittleEndian, 0);
  Result = DA.getU32(&Offset);
  return true;
}

bool DWARFUnit::extractImpl(DataExtractor debug_info, uint32_t *offset_ptr) {
  Length = debug_info.getU32(offset_ptr);
  Version = debug_info.getU16(offset_ptr);
  uint64_t AbbrOffset = debug_info.getU32(offset_ptr);
  AddrSize = debug_info.getU8(offset_ptr);

  bool LengthOK = debug_info.isValidOffset(getNextUnitOffset() - 1);
  bool VersionOK = DWARFContext::isSupportedVersion(Version);
  bool AddrSizeOK = AddrSize == 4 || AddrSize == 8;

  if (!LengthOK || !VersionOK || !AddrSizeOK)
    return false;

  Abbrevs = Abbrev->getAbbreviationDeclarationSet(AbbrOffset);
  return Abbrevs != nullptr;
}

bool DWARFUnit::extract(DataExtractor debug_info, uint32_t *offset_ptr) {
  clear();

  Offset = *offset_ptr;

  if (debug_info.isValidOffset(*offset_ptr)) {
    if (extractImpl(debug_info, offset_ptr))
      return true;

    // reset the offset to where we tried to parse from if anything went wrong
    *offset_ptr = Offset;
  }

  return false;
}

bool DWARFUnit::extractRangeList(uint32_t RangeListOffset,
                                        DWARFDebugRangeList &RangeList) const {
  // Require that compile unit is extracted.
  assert(DieArray.size() > 0);
  DataExtractor RangesData(RangeSection, isLittleEndian, AddrSize);
  uint32_t ActualRangeListOffset = RangeSectionBase + RangeListOffset;
  return RangeList.extract(RangesData, &ActualRangeListOffset);
}

void DWARFUnit::clear() {
  Offset = 0;
  Length = 0;
  Version = 0;
  Abbrevs = nullptr;
  AddrSize = 0;
  BaseAddr = 0;
  RangeSectionBase = 0;
  AddrOffsetSectionBase = 0;
  clearDIEs(false);
  DWO.reset();
}

const char *DWARFUnit::getCompilationDir() {
  extractDIEsIfNeeded(true);
  if (DieArray.empty())
    return nullptr;
  return DieArray[0].getAttributeValueAsString(this, DW_AT_comp_dir, nullptr);
}

uint64_t DWARFUnit::getDWOId() {
  extractDIEsIfNeeded(true);
  const uint64_t FailValue = -1ULL;
  if (DieArray.empty())
    return FailValue;
  return DieArray[0]
      .getAttributeValueAsUnsignedConstant(this, DW_AT_GNU_dwo_id, FailValue);
}

void DWARFUnit::setDIERelations() {
  if (DieArray.size() <= 1)
    return;

  std::vector<DWARFDebugInfoEntryMinimal *> ParentChain;
  DWARFDebugInfoEntryMinimal *SiblingChain = nullptr;
  for (auto &DIE : DieArray) {
    if (SiblingChain) {
      SiblingChain->setSibling(&DIE);
    }
    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            DIE.getAbbreviationDeclarationPtr()) {
      // Normal DIE.
      if (AbbrDecl->hasChildren()) {
        ParentChain.push_back(&DIE);
        SiblingChain = nullptr;
      } else {
        SiblingChain = &DIE;
      }
    } else {
      // NULL entry terminates the sibling chain.
      SiblingChain = ParentChain.back();
      ParentChain.pop_back();
    }
  }
  assert(SiblingChain == nullptr || SiblingChain == &DieArray[0]);
  assert(ParentChain.empty());
}

void DWARFUnit::extractDIEsToVector(
    bool AppendCUDie, bool AppendNonCUDies,
    std::vector<DWARFDebugInfoEntryMinimal> &Dies) const {
  if (!AppendCUDie && !AppendNonCUDies)
    return;

  // Set the offset to that of the first DIE and calculate the start of the
  // next compilation unit header.
  uint32_t DIEOffset = Offset + getHeaderSize();
  uint32_t NextCUOffset = getNextUnitOffset();
  DWARFDebugInfoEntryMinimal DIE;
  uint32_t Depth = 0;
  bool IsCUDie = true;

  while (DIEOffset < NextCUOffset && DIE.extractFast(this, &DIEOffset)) {
    if (IsCUDie) {
      if (AppendCUDie)
        Dies.push_back(DIE);
      if (!AppendNonCUDies)
        break;
      // The average bytes per DIE entry has been seen to be
      // around 14-20 so let's pre-reserve the needed memory for
      // our DIE entries accordingly.
      Dies.reserve(Dies.size() + getDebugInfoSize() / 14);
      IsCUDie = false;
    } else {
      Dies.push_back(DIE);
    }

    if (const DWARFAbbreviationDeclaration *AbbrDecl =
            DIE.getAbbreviationDeclarationPtr()) {
      // Normal DIE
      if (AbbrDecl->hasChildren())
        ++Depth;
    } else {
      // NULL DIE.
      if (Depth > 0)
        --Depth;
      if (Depth == 0)
        break;  // We are done with this compile unit!
    }
  }

  // Give a little bit of info if we encounter corrupt DWARF (our offset
  // should always terminate at or before the start of the next compilation
  // unit header).
  if (DIEOffset > NextCUOffset)
    fprintf(stderr, "warning: DWARF compile unit extends beyond its "
                    "bounds cu 0x%8.8x at 0x%8.8x'\n", getOffset(), DIEOffset);
}

size_t DWARFUnit::extractDIEsIfNeeded(bool CUDieOnly) {
  if ((CUDieOnly && DieArray.size() > 0) ||
      DieArray.size() > 1)
    return 0; // Already parsed.

  bool HasCUDie = DieArray.size() > 0;
  extractDIEsToVector(!HasCUDie, !CUDieOnly, DieArray);

  if (DieArray.empty())
    return 0;

  // If CU DIE was just parsed, copy several attribute values from it.
  if (!HasCUDie) {
    uint64_t BaseAddr =
        DieArray[0].getAttributeValueAsAddress(this, DW_AT_low_pc, -1ULL);
    if (BaseAddr == -1ULL)
      BaseAddr = DieArray[0].getAttributeValueAsAddress(this, DW_AT_entry_pc, 0);
    setBaseAddress(BaseAddr);
    AddrOffsetSectionBase = DieArray[0].getAttributeValueAsSectionOffset(
        this, DW_AT_GNU_addr_base, 0);
    RangeSectionBase = DieArray[0].getAttributeValueAsSectionOffset(
        this, DW_AT_ranges_base, 0);
    // Don't fall back to DW_AT_GNU_ranges_base: it should be ignored for
    // skeleton CU DIE, so that DWARF users not aware of it are not broken.
  }

  setDIERelations();
  return DieArray.size();
}

DWARFUnit::DWOHolder::DWOHolder(StringRef DWOPath)
    : DWOFile(), DWOContext(), DWOU(nullptr) {
  auto Obj = object::ObjectFile::createObjectFile(DWOPath);
  if (!Obj)
    return;
  DWOFile = std::move(Obj.get());
  DWOContext.reset(
      cast<DWARFContext>(new DWARFContextInMemory(*DWOFile.getBinary())));
  if (DWOContext->getNumDWOCompileUnits() > 0)
    DWOU = DWOContext->getDWOCompileUnitAtIndex(0);
}

bool DWARFUnit::parseDWO() {
  if (DWO.get())
    return false;
  extractDIEsIfNeeded(true);
  if (DieArray.empty())
    return false;
  const char *DWOFileName =
      DieArray[0].getAttributeValueAsString(this, DW_AT_GNU_dwo_name, nullptr);
  if (!DWOFileName)
    return false;
  const char *CompilationDir =
      DieArray[0].getAttributeValueAsString(this, DW_AT_comp_dir, nullptr);
  SmallString<16> AbsolutePath;
  if (sys::path::is_relative(DWOFileName) && CompilationDir != nullptr) {
    sys::path::append(AbsolutePath, CompilationDir);
  }
  sys::path::append(AbsolutePath, DWOFileName);
  DWO = llvm::make_unique<DWOHolder>(AbsolutePath);
  DWARFUnit *DWOCU = DWO->getUnit();
  // Verify that compile unit in .dwo file is valid.
  if (!DWOCU || DWOCU->getDWOId() != getDWOId()) {
    DWO.reset();
    return false;
  }
  // Share .debug_addr and .debug_ranges section with compile unit in .dwo
  DWOCU->setAddrOffsetSection(AddrOffsetSection, AddrOffsetSectionBase);
  uint32_t DWORangesBase = DieArray[0].getRangesBaseAttribute(this, 0);
  DWOCU->setRangesSection(RangeSection, DWORangesBase);
  return true;
}

void DWARFUnit::clearDIEs(bool KeepCUDie) {
  if (DieArray.size() > (unsigned)KeepCUDie) {
    // std::vectors never get any smaller when resized to a smaller size,
    // or when clear() or erase() are called, the size will report that it
    // is smaller, but the memory allocated remains intact (call capacity()
    // to see this). So we need to create a temporary vector and swap the
    // contents which will cause just the internal pointers to be swapped
    // so that when temporary vector goes out of scope, it will destroy the
    // contents.
    std::vector<DWARFDebugInfoEntryMinimal> TmpArray;
    DieArray.swap(TmpArray);
    // Save at least the compile unit DIE
    if (KeepCUDie)
      DieArray.push_back(TmpArray.front());
  }
}

void DWARFUnit::collectAddressRanges(DWARFAddressRangesVector &CURanges) {
  const auto *U = getUnitDIE();
  if (U == nullptr)
    return;
  // First, check if unit DIE describes address ranges for the whole unit.
  const auto &CUDIERanges = U->getAddressRanges(this);
  if (!CUDIERanges.empty()) {
    CURanges.insert(CURanges.end(), CUDIERanges.begin(), CUDIERanges.end());
    return;
  }

  // This function is usually called if there in no .debug_aranges section
  // in order to produce a compile unit level set of address ranges that
  // is accurate. If the DIEs weren't parsed, then we don't want all dies for
  // all compile units to stay loaded when they weren't needed. So we can end
  // up parsing the DWARF and then throwing them all away to keep memory usage
  // down.
  const bool ClearDIEs = extractDIEsIfNeeded(false) > 1;
  DieArray[0].collectChildrenAddressRanges(this, CURanges);

  // Collect address ranges from DIEs in .dwo if necessary.
  bool DWOCreated = parseDWO();
  if (DWO.get())
    DWO->getUnit()->collectAddressRanges(CURanges);
  if (DWOCreated)
    DWO.reset();

  // Keep memory down by clearing DIEs if this generate function
  // caused them to be parsed.
  if (ClearDIEs)
    clearDIEs(true);
}

const DWARFDebugInfoEntryMinimal *
DWARFUnit::getSubprogramForAddress(uint64_t Address) {
  extractDIEsIfNeeded(false);
  for (const DWARFDebugInfoEntryMinimal &DIE : DieArray) {
    if (DIE.isSubprogramDIE() &&
        DIE.addressRangeContainsAddress(this, Address)) {
      return &DIE;
    }
  }
  return nullptr;
}

DWARFDebugInfoEntryInlinedChain
DWARFUnit::getInlinedChainForAddress(uint64_t Address) {
  // First, find a subprogram that contains the given address (the root
  // of inlined chain).
  const DWARFUnit *ChainCU = nullptr;
  const DWARFDebugInfoEntryMinimal *SubprogramDIE =
      getSubprogramForAddress(Address);
  if (SubprogramDIE) {
    ChainCU = this;
  } else {
    // Try to look for subprogram DIEs in the DWO file.
    parseDWO();
    if (DWO.get()) {
      SubprogramDIE = DWO->getUnit()->getSubprogramForAddress(Address);
      if (SubprogramDIE)
        ChainCU = DWO->getUnit();
    }
  }

  // Get inlined chain rooted at this subprogram DIE.
  if (!SubprogramDIE)
    return DWARFDebugInfoEntryInlinedChain();
  return SubprogramDIE->getInlinedChainForAddress(ChainCU, Address);
}
