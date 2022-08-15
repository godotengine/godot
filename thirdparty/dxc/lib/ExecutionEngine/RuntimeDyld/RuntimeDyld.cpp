//===-- RuntimeDyld.cpp - Run-time dynamic linker for MC-JIT ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MC-JIT runtime dynamic linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "RuntimeDyldCheckerImpl.h"
#include "RuntimeDyldCOFF.h"
#include "RuntimeDyldELF.h"
#include "RuntimeDyldImpl.h"
#include "RuntimeDyldMachO.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/COFF.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MutexGuard.h"

using namespace llvm;
using namespace llvm::object;

#define DEBUG_TYPE "dyld"

// Empty out-of-line virtual destructor as the key function.
RuntimeDyldImpl::~RuntimeDyldImpl() {}

// Pin LoadedObjectInfo's vtables to this file.
void RuntimeDyld::LoadedObjectInfo::anchor() {}

namespace llvm {

void RuntimeDyldImpl::registerEHFrames() {}

void RuntimeDyldImpl::deregisterEHFrames() {}

#ifndef NDEBUG
static void dumpSectionMemory(const SectionEntry &S, StringRef State) {
  dbgs() << "----- Contents of section " << S.Name << " " << State << " -----";

  if (S.Address == nullptr) {
    dbgs() << "\n          <section not emitted>\n";
    return;
  }

  const unsigned ColsPerRow = 16;

  uint8_t *DataAddr = S.Address;
  uint64_t LoadAddr = S.LoadAddress;

  unsigned StartPadding = LoadAddr & (ColsPerRow - 1);
  unsigned BytesRemaining = S.Size;

  if (StartPadding) {
    dbgs() << "\n" << format("0x%016" PRIx64,
                             LoadAddr & ~(uint64_t)(ColsPerRow - 1)) << ":";
    while (StartPadding--)
      dbgs() << "   ";
  }

  while (BytesRemaining > 0) {
    if ((LoadAddr & (ColsPerRow - 1)) == 0)
      dbgs() << "\n" << format("0x%016" PRIx64, LoadAddr) << ":";

    dbgs() << " " << format("%02x", *DataAddr);

    ++DataAddr;
    ++LoadAddr;
    --BytesRemaining;
  }

  dbgs() << "\n";
}
#endif

// Resolve the relocations for all symbols we currently know about.
void RuntimeDyldImpl::resolveRelocations() {
  MutexGuard locked(lock);

  // First, resolve relocations associated with external symbols.
  resolveExternalSymbols();

  // Just iterate over the sections we have and resolve all the relocations
  // in them. Gross overkill, but it gets the job done.
  for (int i = 0, e = Sections.size(); i != e; ++i) {
    // The Section here (Sections[i]) refers to the section in which the
    // symbol for the relocation is located.  The SectionID in the relocation
    // entry provides the section to which the relocation will be applied.
    uint64_t Addr = Sections[i].LoadAddress;
    DEBUG(dbgs() << "Resolving relocations Section #" << i << "\t"
                 << format("%p", (uintptr_t)Addr) << "\n");
    DEBUG(dumpSectionMemory(Sections[i], "before relocations"));
    resolveRelocationList(Relocations[i], Addr);
    DEBUG(dumpSectionMemory(Sections[i], "after relocations"));
    Relocations.erase(i);
  }
}

void RuntimeDyldImpl::mapSectionAddress(const void *LocalAddress,
                                        uint64_t TargetAddress) {
  MutexGuard locked(lock);
  for (unsigned i = 0, e = Sections.size(); i != e; ++i) {
    if (Sections[i].Address == LocalAddress) {
      reassignSectionAddress(i, TargetAddress);
      return;
    }
  }
  llvm_unreachable("Attempting to remap address of unknown section!");
}

static std::error_code getOffset(const SymbolRef &Sym, SectionRef Sec,
                                 uint64_t &Result) {
  ErrorOr<uint64_t> AddressOrErr = Sym.getAddress();
  if (std::error_code EC = AddressOrErr.getError())
    return EC;
  Result = *AddressOrErr - Sec.getAddress();
  return std::error_code();
}

std::pair<unsigned, unsigned>
RuntimeDyldImpl::loadObjectImpl(const object::ObjectFile &Obj) {
  MutexGuard locked(lock);

  // Grab the first Section ID. We'll use this later to construct the underlying
  // range for the returned LoadedObjectInfo.
  unsigned SectionsAddedBeginIdx = Sections.size();

  // Save information about our target
  Arch = (Triple::ArchType)Obj.getArch();
  IsTargetLittleEndian = Obj.isLittleEndian();
  setMipsABI(Obj);

  // Compute the memory size required to load all sections to be loaded
  // and pass this information to the memory manager
  if (MemMgr.needsToReserveAllocationSpace()) {
    uint64_t CodeSize = 0, DataSizeRO = 0, DataSizeRW = 0;
    computeTotalAllocSize(Obj, CodeSize, DataSizeRO, DataSizeRW);
    MemMgr.reserveAllocationSpace(CodeSize, DataSizeRO, DataSizeRW);
  }

  // Used sections from the object file
  ObjSectionToIDMap LocalSections;

  // Common symbols requiring allocation, with their sizes and alignments
  CommonSymbolList CommonSymbols;

  // Parse symbols
  DEBUG(dbgs() << "Parse symbols:\n");
  for (symbol_iterator I = Obj.symbol_begin(), E = Obj.symbol_end(); I != E;
       ++I) {
    uint32_t Flags = I->getFlags();

    bool IsCommon = Flags & SymbolRef::SF_Common;
    if (IsCommon)
      CommonSymbols.push_back(*I);
    else {
      object::SymbolRef::Type SymType = I->getType();

      if (SymType == object::SymbolRef::ST_Function ||
          SymType == object::SymbolRef::ST_Data ||
          SymType == object::SymbolRef::ST_Unknown) {

        ErrorOr<StringRef> NameOrErr = I->getName();
        Check(NameOrErr.getError());
        StringRef Name = *NameOrErr;
        section_iterator SI = Obj.section_end();
        Check(I->getSection(SI));
        if (SI == Obj.section_end())
          continue;
        uint64_t SectOffset;
        Check(getOffset(*I, *SI, SectOffset));
        StringRef SectionData;
        Check(SI->getContents(SectionData));
        bool IsCode = SI->isText();
        unsigned SectionID =
            findOrEmitSection(Obj, *SI, IsCode, LocalSections);
        DEBUG(dbgs() << "\tType: " << SymType << " Name: " << Name
                     << " SID: " << SectionID << " Offset: "
                     << format("%p", (uintptr_t)SectOffset)
                     << " flags: " << Flags << "\n");
        JITSymbolFlags RTDyldSymFlags = JITSymbolFlags::None;
        if (Flags & SymbolRef::SF_Weak)
          RTDyldSymFlags |= JITSymbolFlags::Weak;
        if (Flags & SymbolRef::SF_Exported)
          RTDyldSymFlags |= JITSymbolFlags::Exported;
        GlobalSymbolTable[Name] =
          SymbolTableEntry(SectionID, SectOffset, RTDyldSymFlags);
      }
    }
  }

  // Allocate common symbols
  emitCommonSymbols(Obj, CommonSymbols);

  // Parse and process relocations
  DEBUG(dbgs() << "Parse relocations:\n");
  for (section_iterator SI = Obj.section_begin(), SE = Obj.section_end();
       SI != SE; ++SI) {
    unsigned SectionID = 0;
    StubMap Stubs;
    section_iterator RelocatedSection = SI->getRelocatedSection();

    if (RelocatedSection == SE)
      continue;

    relocation_iterator I = SI->relocation_begin();
    relocation_iterator E = SI->relocation_end();

    if (I == E && !ProcessAllSections)
      continue;

    bool IsCode = RelocatedSection->isText();
    SectionID =
        findOrEmitSection(Obj, *RelocatedSection, IsCode, LocalSections);
    DEBUG(dbgs() << "\tSectionID: " << SectionID << "\n");

    for (; I != E;)
      I = processRelocationRef(SectionID, I, Obj, LocalSections, Stubs);

    // If there is an attached checker, notify it about the stubs for this
    // section so that they can be verified.
    if (Checker)
      Checker->registerStubMap(Obj.getFileName(), SectionID, Stubs);
  }

  // Give the subclasses a chance to tie-up any loose ends.
  finalizeLoad(Obj, LocalSections);

  unsigned SectionsAddedEndIdx = Sections.size();

  return std::make_pair(SectionsAddedBeginIdx, SectionsAddedEndIdx);
}

// A helper method for computeTotalAllocSize.
// Computes the memory size required to allocate sections with the given sizes,
// assuming that all sections are allocated with the given alignment
static uint64_t
computeAllocationSizeForSections(std::vector<uint64_t> &SectionSizes,
                                 uint64_t Alignment) {
  uint64_t TotalSize = 0;
  for (size_t Idx = 0, Cnt = SectionSizes.size(); Idx < Cnt; Idx++) {
    uint64_t AlignedSize =
        (SectionSizes[Idx] + Alignment - 1) / Alignment * Alignment;
    TotalSize += AlignedSize;
  }
  return TotalSize;
}

static bool isRequiredForExecution(const SectionRef Section) {
  const ObjectFile *Obj = Section.getObject();
  if (isa<object::ELFObjectFileBase>(Obj))
    return ELFSectionRef(Section).getFlags() & ELF::SHF_ALLOC;
  if (auto *COFFObj = dyn_cast<object::COFFObjectFile>(Obj)) {
    const coff_section *CoffSection = COFFObj->getCOFFSection(Section);
    // Avoid loading zero-sized COFF sections.
    // In PE files, VirtualSize gives the section size, and SizeOfRawData
    // may be zero for sections with content. In Obj files, SizeOfRawData 
    // gives the section size, and VirtualSize is always zero. Hence
    // the need to check for both cases below.
    bool HasContent = (CoffSection->VirtualSize > 0) 
      || (CoffSection->SizeOfRawData > 0);
    bool IsDiscardable = CoffSection->Characteristics &
      (COFF::IMAGE_SCN_MEM_DISCARDABLE | COFF::IMAGE_SCN_LNK_INFO);
    return HasContent && !IsDiscardable;
  }
  
  assert(isa<MachOObjectFile>(Obj));
  return true;
}

static bool isReadOnlyData(const SectionRef Section) {
  const ObjectFile *Obj = Section.getObject();
  if (isa<object::ELFObjectFileBase>(Obj))
    return !(ELFSectionRef(Section).getFlags() &
             (ELF::SHF_WRITE | ELF::SHF_EXECINSTR));
  if (auto *COFFObj = dyn_cast<object::COFFObjectFile>(Obj))
    return ((COFFObj->getCOFFSection(Section)->Characteristics &
             (COFF::IMAGE_SCN_CNT_INITIALIZED_DATA
             | COFF::IMAGE_SCN_MEM_READ
             | COFF::IMAGE_SCN_MEM_WRITE))
             ==
             (COFF::IMAGE_SCN_CNT_INITIALIZED_DATA
             | COFF::IMAGE_SCN_MEM_READ));

  assert(isa<MachOObjectFile>(Obj));
  return false;
}

static bool isZeroInit(const SectionRef Section) {
  const ObjectFile *Obj = Section.getObject();
  if (isa<object::ELFObjectFileBase>(Obj))
    return ELFSectionRef(Section).getType() == ELF::SHT_NOBITS;
  if (auto *COFFObj = dyn_cast<object::COFFObjectFile>(Obj))
    return COFFObj->getCOFFSection(Section)->Characteristics &
            COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;

  auto *MachO = cast<MachOObjectFile>(Obj);
  unsigned SectionType = MachO->getSectionType(Section);
  return SectionType == MachO::S_ZEROFILL ||
         SectionType == MachO::S_GB_ZEROFILL;
}

// Compute an upper bound of the memory size that is required to load all
// sections
void RuntimeDyldImpl::computeTotalAllocSize(const ObjectFile &Obj,
                                            uint64_t &CodeSize,
                                            uint64_t &DataSizeRO,
                                            uint64_t &DataSizeRW) {
  // Compute the size of all sections required for execution
  std::vector<uint64_t> CodeSectionSizes;
  std::vector<uint64_t> ROSectionSizes;
  std::vector<uint64_t> RWSectionSizes;
  uint64_t MaxAlignment = sizeof(void *);

  // Collect sizes of all sections to be loaded;
  // also determine the max alignment of all sections
  for (section_iterator SI = Obj.section_begin(), SE = Obj.section_end();
       SI != SE; ++SI) {
    const SectionRef &Section = *SI;

    bool IsRequired = isRequiredForExecution(Section);

    // Consider only the sections that are required to be loaded for execution
    if (IsRequired) {
      StringRef Name;
      uint64_t DataSize = Section.getSize();
      uint64_t Alignment64 = Section.getAlignment();
      bool IsCode = Section.isText();
      bool IsReadOnly = isReadOnlyData(Section);
      Check(Section.getName(Name));
      unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;

      uint64_t StubBufSize = computeSectionStubBufSize(Obj, Section);
      uint64_t SectionSize = DataSize + StubBufSize;

      // The .eh_frame section (at least on Linux) needs an extra four bytes
      // padded
      // with zeroes added at the end.  For MachO objects, this section has a
      // slightly different name, so this won't have any effect for MachO
      // objects.
      if (Name == ".eh_frame")
        SectionSize += 4;

      if (!SectionSize)
        SectionSize = 1;

      if (IsCode) {
        CodeSectionSizes.push_back(SectionSize);
      } else if (IsReadOnly) {
        ROSectionSizes.push_back(SectionSize);
      } else {
        RWSectionSizes.push_back(SectionSize);
      }

      // update the max alignment
      if (Alignment > MaxAlignment) {
        MaxAlignment = Alignment;
      }
    }
  }

  // Compute the size of all common symbols
  uint64_t CommonSize = 0;
  for (symbol_iterator I = Obj.symbol_begin(), E = Obj.symbol_end(); I != E;
       ++I) {
    uint32_t Flags = I->getFlags();
    if (Flags & SymbolRef::SF_Common) {
      // Add the common symbols to a list.  We'll allocate them all below.
      uint64_t Size = I->getCommonSize();
      CommonSize += Size;
    }
  }
  if (CommonSize != 0) {
    RWSectionSizes.push_back(CommonSize);
  }

  // Compute the required allocation space for each different type of sections
  // (code, read-only data, read-write data) assuming that all sections are
  // allocated with the max alignment. Note that we cannot compute with the
  // individual alignments of the sections, because then the required size
  // depends on the order, in which the sections are allocated.
  CodeSize = computeAllocationSizeForSections(CodeSectionSizes, MaxAlignment);
  DataSizeRO = computeAllocationSizeForSections(ROSectionSizes, MaxAlignment);
  DataSizeRW = computeAllocationSizeForSections(RWSectionSizes, MaxAlignment);
}

// compute stub buffer size for the given section
unsigned RuntimeDyldImpl::computeSectionStubBufSize(const ObjectFile &Obj,
                                                    const SectionRef &Section) {
  unsigned StubSize = getMaxStubSize();
  if (StubSize == 0) {
    return 0;
  }
  // FIXME: this is an inefficient way to handle this. We should computed the
  // necessary section allocation size in loadObject by walking all the sections
  // once.
  unsigned StubBufSize = 0;
  for (section_iterator SI = Obj.section_begin(), SE = Obj.section_end();
       SI != SE; ++SI) {
    section_iterator RelSecI = SI->getRelocatedSection();
    if (!(RelSecI == Section))
      continue;

    for (const RelocationRef &Reloc : SI->relocations()) {
      (void)Reloc;
      StubBufSize += StubSize;
    }
  }

  // Get section data size and alignment
  uint64_t DataSize = Section.getSize();
  uint64_t Alignment64 = Section.getAlignment();

  // Add stubbuf size alignment
  unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;
  unsigned StubAlignment = getStubAlignment();
  unsigned EndAlignment = (DataSize | Alignment) & -(DataSize | Alignment);
  if (StubAlignment > EndAlignment)
    StubBufSize += StubAlignment - EndAlignment;
  return StubBufSize;
}

uint64_t RuntimeDyldImpl::readBytesUnaligned(uint8_t *Src,
                                             unsigned Size) const {
  uint64_t Result = 0;
  if (IsTargetLittleEndian) {
    Src += Size - 1;
    while (Size--)
      Result = (Result << 8) | *Src--;
  } else
    while (Size--)
      Result = (Result << 8) | *Src++;

  return Result;
}

void RuntimeDyldImpl::writeBytesUnaligned(uint64_t Value, uint8_t *Dst,
                                          unsigned Size) const {
  if (IsTargetLittleEndian) {
    while (Size--) {
      *Dst++ = Value & 0xFF;
      Value >>= 8;
    }
  } else {
    Dst += Size - 1;
    while (Size--) {
      *Dst-- = Value & 0xFF;
      Value >>= 8;
    }
  }
}

void RuntimeDyldImpl::emitCommonSymbols(const ObjectFile &Obj,
                                        CommonSymbolList &CommonSymbols) {
  if (CommonSymbols.empty())
    return;

  uint64_t CommonSize = 0;
  CommonSymbolList SymbolsToAllocate;

  DEBUG(dbgs() << "Processing common symbols...\n");

  for (const auto &Sym : CommonSymbols) {
    ErrorOr<StringRef> NameOrErr = Sym.getName();
    Check(NameOrErr.getError());
    StringRef Name = *NameOrErr;

    // Skip common symbols already elsewhere.
    if (GlobalSymbolTable.count(Name) ||
        Resolver.findSymbolInLogicalDylib(Name)) {
      DEBUG(dbgs() << "\tSkipping already emitted common symbol '" << Name
                   << "'\n");
      continue;
    }

    uint32_t Align = Sym.getAlignment();
    uint64_t Size = Sym.getCommonSize();

    CommonSize += Align + Size;
    SymbolsToAllocate.push_back(Sym);
  }

  // Allocate memory for the section
  unsigned SectionID = Sections.size();
  uint8_t *Addr = MemMgr.allocateDataSection(CommonSize, sizeof(void *),
                                             SectionID, StringRef(), false);
  if (!Addr)
    report_fatal_error("Unable to allocate memory for common symbols!");
  uint64_t Offset = 0;
  Sections.push_back(SectionEntry("<common symbols>", Addr, CommonSize, 0));
  memset(Addr, 0, CommonSize);

  DEBUG(dbgs() << "emitCommonSection SectionID: " << SectionID << " new addr: "
               << format("%p", Addr) << " DataSize: " << CommonSize << "\n");

  // Assign the address of each symbol
  for (auto &Sym : SymbolsToAllocate) {
    uint32_t Align = Sym.getAlignment();
    uint64_t Size = Sym.getCommonSize();
    ErrorOr<StringRef> NameOrErr = Sym.getName();
    Check(NameOrErr.getError());
    StringRef Name = *NameOrErr;
    if (Align) {
      // This symbol has an alignment requirement.
      uint64_t AlignOffset = OffsetToAlignment((uint64_t)Addr, Align);
      Addr += AlignOffset;
      Offset += AlignOffset;
    }
    uint32_t Flags = Sym.getFlags();
    JITSymbolFlags RTDyldSymFlags = JITSymbolFlags::None;
    if (Flags & SymbolRef::SF_Weak)
      RTDyldSymFlags |= JITSymbolFlags::Weak;
    if (Flags & SymbolRef::SF_Exported)
      RTDyldSymFlags |= JITSymbolFlags::Exported;
    DEBUG(dbgs() << "Allocating common symbol " << Name << " address "
                 << format("%p", Addr) << "\n");
    GlobalSymbolTable[Name] =
      SymbolTableEntry(SectionID, Offset, RTDyldSymFlags);
    Offset += Size;
    Addr += Size;
  }
}

unsigned RuntimeDyldImpl::emitSection(const ObjectFile &Obj,
                                      const SectionRef &Section, bool IsCode) {

  StringRef data;
  uint64_t Alignment64 = Section.getAlignment();

  unsigned Alignment = (unsigned)Alignment64 & 0xffffffffL;
  unsigned PaddingSize = 0;
  unsigned StubBufSize = 0;
  StringRef Name;
  bool IsRequired = isRequiredForExecution(Section);
  bool IsVirtual = Section.isVirtual();
  bool IsZeroInit = isZeroInit(Section);
  bool IsReadOnly = isReadOnlyData(Section);
  uint64_t DataSize = Section.getSize();
  Check(Section.getName(Name));

  StubBufSize = computeSectionStubBufSize(Obj, Section);

  // The .eh_frame section (at least on Linux) needs an extra four bytes padded
  // with zeroes added at the end.  For MachO objects, this section has a
  // slightly different name, so this won't have any effect for MachO objects.
  if (Name == ".eh_frame")
    PaddingSize = 4;

  uintptr_t Allocate;
  unsigned SectionID = Sections.size();
  uint8_t *Addr;
  const char *pData = nullptr;

  // In either case, set the location of the unrelocated section in memory,
  // since we still process relocations for it even if we're not applying them.
  Check(Section.getContents(data));
  // Virtual sections have no data in the object image, so leave pData = 0
  if (!IsVirtual)
    pData = data.data();

  // Some sections, such as debug info, don't need to be loaded for execution.
  // Leave those where they are.
  if (IsRequired) {
    Allocate = DataSize + PaddingSize + StubBufSize;
    if (!Allocate)
      Allocate = 1;
    Addr = IsCode ? MemMgr.allocateCodeSection(Allocate, Alignment, SectionID,
                                               Name)
                  : MemMgr.allocateDataSection(Allocate, Alignment, SectionID,
                                               Name, IsReadOnly);
    if (!Addr)
      report_fatal_error("Unable to allocate section memory!");

    // Zero-initialize or copy the data from the image
    if (IsZeroInit || IsVirtual)
      memset(Addr, 0, DataSize);
    else
      memcpy(Addr, pData, DataSize);

    // Fill in any extra bytes we allocated for padding
    if (PaddingSize != 0) {
      memset(Addr + DataSize, 0, PaddingSize);
      // Update the DataSize variable so that the stub offset is set correctly.
      DataSize += PaddingSize;
    }

    DEBUG(dbgs() << "emitSection SectionID: " << SectionID << " Name: " << Name
                 << " obj addr: " << format("%p", pData)
                 << " new addr: " << format("%p", Addr)
                 << " DataSize: " << DataSize << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate << "\n");
  } else {
    // Even if we didn't load the section, we need to record an entry for it
    // to handle later processing (and by 'handle' I mean don't do anything
    // with these sections).
    Allocate = 0;
    Addr = nullptr;
    DEBUG(dbgs() << "emitSection SectionID: " << SectionID << " Name: " << Name
                 << " obj addr: " << format("%p", data.data()) << " new addr: 0"
                 << " DataSize: " << DataSize << " StubBufSize: " << StubBufSize
                 << " Allocate: " << Allocate << "\n");
  }

  Sections.push_back(SectionEntry(Name, Addr, DataSize, (uintptr_t)pData));

  if (Checker)
    Checker->registerSection(Obj.getFileName(), SectionID);

  return SectionID;
}

unsigned RuntimeDyldImpl::findOrEmitSection(const ObjectFile &Obj,
                                            const SectionRef &Section,
                                            bool IsCode,
                                            ObjSectionToIDMap &LocalSections) {

  unsigned SectionID = 0;
  ObjSectionToIDMap::iterator i = LocalSections.find(Section);
  if (i != LocalSections.end())
    SectionID = i->second;
  else {
    SectionID = emitSection(Obj, Section, IsCode);
    LocalSections[Section] = SectionID;
  }
  return SectionID;
}

void RuntimeDyldImpl::addRelocationForSection(const RelocationEntry &RE,
                                              unsigned SectionID) {
  Relocations[SectionID].push_back(RE);
}

void RuntimeDyldImpl::addRelocationForSymbol(const RelocationEntry &RE,
                                             StringRef SymbolName) {
  // Relocation by symbol.  If the symbol is found in the global symbol table,
  // create an appropriate section relocation.  Otherwise, add it to
  // ExternalSymbolRelocations.
  RTDyldSymbolTable::const_iterator Loc = GlobalSymbolTable.find(SymbolName);
  if (Loc == GlobalSymbolTable.end()) {
    ExternalSymbolRelocations[SymbolName].push_back(RE);
  } else {
    // Copy the RE since we want to modify its addend.
    RelocationEntry RECopy = RE;
    const auto &SymInfo = Loc->second;
    RECopy.Addend += SymInfo.getOffset();
    Relocations[SymInfo.getSectionID()].push_back(RECopy);
  }
}

uint8_t *RuntimeDyldImpl::createStubFunction(uint8_t *Addr,
                                             unsigned AbiVariant) {
  if (Arch == Triple::aarch64 || Arch == Triple::aarch64_be) {
    // This stub has to be able to access the full address space,
    // since symbol lookup won't necessarily find a handy, in-range,
    // PLT stub for functions which could be anywhere.
    // Stub can use ip0 (== x16) to calculate address
    writeBytesUnaligned(0xd2e00010, Addr,    4); // movz ip0, #:abs_g3:<addr>
    writeBytesUnaligned(0xf2c00010, Addr+4,  4); // movk ip0, #:abs_g2_nc:<addr>
    writeBytesUnaligned(0xf2a00010, Addr+8,  4); // movk ip0, #:abs_g1_nc:<addr>
    writeBytesUnaligned(0xf2800010, Addr+12, 4); // movk ip0, #:abs_g0_nc:<addr>
    writeBytesUnaligned(0xd61f0200, Addr+16, 4); // br ip0

    return Addr;
  } else if (Arch == Triple::arm || Arch == Triple::armeb) {
    // TODO: There is only ARM far stub now. We should add the Thumb stub,
    // and stubs for branches Thumb - ARM and ARM - Thumb.
    writeBytesUnaligned(0xe51ff004, Addr, 4); // ldr pc,<label>
    return Addr + 4;
  } else if (IsMipsO32ABI) {
    // 0:   3c190000        lui     t9,%hi(addr).
    // 4:   27390000        addiu   t9,t9,%lo(addr).
    // 8:   03200008        jr      t9.
    // c:   00000000        nop.
    const unsigned LuiT9Instr = 0x3c190000, AdduiT9Instr = 0x27390000;
    const unsigned JrT9Instr = 0x03200008, NopInstr = 0x0;

    writeBytesUnaligned(LuiT9Instr, Addr, 4);
    writeBytesUnaligned(AdduiT9Instr, Addr+4, 4);
    writeBytesUnaligned(JrT9Instr, Addr+8, 4);
    writeBytesUnaligned(NopInstr, Addr+12, 4);
    return Addr;
  } else if (Arch == Triple::ppc64 || Arch == Triple::ppc64le) {
    // Depending on which version of the ELF ABI is in use, we need to
    // generate one of two variants of the stub.  They both start with
    // the same sequence to load the target address into r12.
    writeInt32BE(Addr,    0x3D800000); // lis   r12, highest(addr)
    writeInt32BE(Addr+4,  0x618C0000); // ori   r12, higher(addr)
    writeInt32BE(Addr+8,  0x798C07C6); // sldi  r12, r12, 32
    writeInt32BE(Addr+12, 0x658C0000); // oris  r12, r12, h(addr)
    writeInt32BE(Addr+16, 0x618C0000); // ori   r12, r12, l(addr)
    if (AbiVariant == 2) {
      // PowerPC64 stub ELFv2 ABI: The address points to the function itself.
      // The address is already in r12 as required by the ABI.  Branch to it.
      writeInt32BE(Addr+20, 0xF8410018); // std   r2,  24(r1)
      writeInt32BE(Addr+24, 0x7D8903A6); // mtctr r12
      writeInt32BE(Addr+28, 0x4E800420); // bctr
    } else {
      // PowerPC64 stub ELFv1 ABI: The address points to a function descriptor.
      // Load the function address on r11 and sets it to control register. Also
      // loads the function TOC in r2 and environment pointer to r11.
      writeInt32BE(Addr+20, 0xF8410028); // std   r2,  40(r1)
      writeInt32BE(Addr+24, 0xE96C0000); // ld    r11, 0(r12)
      writeInt32BE(Addr+28, 0xE84C0008); // ld    r2,  0(r12)
      writeInt32BE(Addr+32, 0x7D6903A6); // mtctr r11
      writeInt32BE(Addr+36, 0xE96C0010); // ld    r11, 16(r2)
      writeInt32BE(Addr+40, 0x4E800420); // bctr
    }
    return Addr;
  } else if (Arch == Triple::systemz) {
    writeInt16BE(Addr,    0xC418);     // lgrl %r1,.+8
    writeInt16BE(Addr+2,  0x0000);
    writeInt16BE(Addr+4,  0x0004);
    writeInt16BE(Addr+6,  0x07F1);     // brc 15,%r1
    // 8-byte address stored at Addr + 8
    return Addr;
  } else if (Arch == Triple::x86_64) {
    *Addr      = 0xFF; // jmp
    *(Addr+1)  = 0x25; // rip
    // 32-bit PC-relative address of the GOT entry will be stored at Addr+2
  } else if (Arch == Triple::x86) {
    *Addr      = 0xE9; // 32-bit pc-relative jump.
  }
  return Addr;
}

// Assign an address to a symbol name and resolve all the relocations
// associated with it.
void RuntimeDyldImpl::reassignSectionAddress(unsigned SectionID,
                                             uint64_t Addr) {
  // The address to use for relocation resolution is not
  // the address of the local section buffer. We must be doing
  // a remote execution environment of some sort. Relocations can't
  // be applied until all the sections have been moved.  The client must
  // trigger this with a call to MCJIT::finalize() or
  // RuntimeDyld::resolveRelocations().
  //
  // Addr is a uint64_t because we can't assume the pointer width
  // of the target is the same as that of the host. Just use a generic
  // "big enough" type.
  DEBUG(dbgs() << "Reassigning address for section "
               << SectionID << " (" << Sections[SectionID].Name << "): "
               << format("0x%016" PRIx64, Sections[SectionID].LoadAddress) << " -> "
               << format("0x%016" PRIx64, Addr) << "\n");
  Sections[SectionID].LoadAddress = Addr;
}

void RuntimeDyldImpl::resolveRelocationList(const RelocationList &Relocs,
                                            uint64_t Value) {
  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    const RelocationEntry &RE = Relocs[i];
    // Ignore relocations for sections that were not loaded
    if (Sections[RE.SectionID].Address == nullptr)
      continue;
    resolveRelocation(RE, Value);
  }
}

void RuntimeDyldImpl::resolveExternalSymbols() {
  while (!ExternalSymbolRelocations.empty()) {
    StringMap<RelocationList>::iterator i = ExternalSymbolRelocations.begin();

    StringRef Name = i->first();
    if (Name.size() == 0) {
      // This is an absolute symbol, use an address of zero.
      DEBUG(dbgs() << "Resolving absolute relocations."
                   << "\n");
      RelocationList &Relocs = i->second;
      resolveRelocationList(Relocs, 0);
    } else {
      uint64_t Addr = 0;
      RTDyldSymbolTable::const_iterator Loc = GlobalSymbolTable.find(Name);
      if (Loc == GlobalSymbolTable.end()) {
        // This is an external symbol, try to get its address from the symbol
        // resolver.
        Addr = Resolver.findSymbol(Name.data()).getAddress();
        // The call to getSymbolAddress may have caused additional modules to
        // be loaded, which may have added new entries to the
        // ExternalSymbolRelocations map.  Consquently, we need to update our
        // iterator.  This is also why retrieval of the relocation list
        // associated with this symbol is deferred until below this point.
        // New entries may have been added to the relocation list.
        i = ExternalSymbolRelocations.find(Name);
      } else {
        // We found the symbol in our global table.  It was probably in a
        // Module that we loaded previously.
        const auto &SymInfo = Loc->second;
        Addr = getSectionLoadAddress(SymInfo.getSectionID()) +
               SymInfo.getOffset();
      }

      // FIXME: Implement error handling that doesn't kill the host program!
      if (!Addr)
        report_fatal_error("Program used external function '" + Name +
                           "' which could not be resolved!");

      // If Resolver returned UINT64_MAX, the client wants to handle this symbol
      // manually and we shouldn't resolve its relocations.
      if (Addr != UINT64_MAX) {
        DEBUG(dbgs() << "Resolving relocations Name: " << Name << "\t"
                     << format("0x%lx", Addr) << "\n");
        // This list may have been updated when we called getSymbolAddress, so
        // don't change this code to get the list earlier.
        RelocationList &Relocs = i->second;
        resolveRelocationList(Relocs, Addr);
      }
    }

    ExternalSymbolRelocations.erase(i);
  }
}

//===----------------------------------------------------------------------===//
// RuntimeDyld class implementation

uint64_t RuntimeDyld::LoadedObjectInfo::getSectionLoadAddress(
                                                  StringRef SectionName) const {
  for (unsigned I = BeginIdx; I != EndIdx; ++I)
    if (RTDyld.Sections[I].Name == SectionName)
      return RTDyld.Sections[I].LoadAddress;

  return 0;
}

void RuntimeDyld::MemoryManager::anchor() {}
void RuntimeDyld::SymbolResolver::anchor() {}

RuntimeDyld::RuntimeDyld(RuntimeDyld::MemoryManager &MemMgr,
                         RuntimeDyld::SymbolResolver &Resolver)
    : MemMgr(MemMgr), Resolver(Resolver) {
  // FIXME: There's a potential issue lurking here if a single instance of
  // RuntimeDyld is used to load multiple objects.  The current implementation
  // associates a single memory manager with a RuntimeDyld instance.  Even
  // though the public class spawns a new 'impl' instance for each load,
  // they share a single memory manager.  This can become a problem when page
  // permissions are applied.
  Dyld = nullptr;
  ProcessAllSections = false;
  Checker = nullptr;
}

RuntimeDyld::~RuntimeDyld() {}

static std::unique_ptr<RuntimeDyldCOFF>
createRuntimeDyldCOFF(Triple::ArchType Arch, RuntimeDyld::MemoryManager &MM,
                      RuntimeDyld::SymbolResolver &Resolver,
                      bool ProcessAllSections, RuntimeDyldCheckerImpl *Checker) {
  std::unique_ptr<RuntimeDyldCOFF> Dyld =
    RuntimeDyldCOFF::create(Arch, MM, Resolver);
  Dyld->setProcessAllSections(ProcessAllSections);
  Dyld->setRuntimeDyldChecker(Checker);
  return Dyld;
}

static std::unique_ptr<RuntimeDyldELF>
createRuntimeDyldELF(RuntimeDyld::MemoryManager &MM,
                     RuntimeDyld::SymbolResolver &Resolver,
                     bool ProcessAllSections, RuntimeDyldCheckerImpl *Checker) {
  std::unique_ptr<RuntimeDyldELF> Dyld(new RuntimeDyldELF(MM, Resolver));
  Dyld->setProcessAllSections(ProcessAllSections);
  Dyld->setRuntimeDyldChecker(Checker);
  return Dyld;
}

static std::unique_ptr<RuntimeDyldMachO>
createRuntimeDyldMachO(Triple::ArchType Arch, RuntimeDyld::MemoryManager &MM,
                       RuntimeDyld::SymbolResolver &Resolver,
                       bool ProcessAllSections,
                       RuntimeDyldCheckerImpl *Checker) {
  std::unique_ptr<RuntimeDyldMachO> Dyld =
    RuntimeDyldMachO::create(Arch, MM, Resolver);
  Dyld->setProcessAllSections(ProcessAllSections);
  Dyld->setRuntimeDyldChecker(Checker);
  return Dyld;
}

std::unique_ptr<RuntimeDyld::LoadedObjectInfo>
RuntimeDyld::loadObject(const ObjectFile &Obj) {
  if (!Dyld) {
    if (Obj.isELF())
      Dyld = createRuntimeDyldELF(MemMgr, Resolver, ProcessAllSections, Checker);
    else if (Obj.isMachO())
      Dyld = createRuntimeDyldMachO(
               static_cast<Triple::ArchType>(Obj.getArch()), MemMgr, Resolver,
               ProcessAllSections, Checker);
    else if (Obj.isCOFF())
      Dyld = createRuntimeDyldCOFF(
               static_cast<Triple::ArchType>(Obj.getArch()), MemMgr, Resolver,
               ProcessAllSections, Checker);
    else
      report_fatal_error("Incompatible object format!");
  }

  if (!Dyld->isCompatibleFile(Obj))
    report_fatal_error("Incompatible object format!");

  return Dyld->loadObject(Obj);
}

void *RuntimeDyld::getSymbolLocalAddress(StringRef Name) const {
  if (!Dyld)
    return nullptr;
  return Dyld->getSymbolLocalAddress(Name);
}

RuntimeDyld::SymbolInfo RuntimeDyld::getSymbol(StringRef Name) const {
  if (!Dyld)
    return nullptr;
  return Dyld->getSymbol(Name);
}

void RuntimeDyld::resolveRelocations() { Dyld->resolveRelocations(); }

void RuntimeDyld::reassignSectionAddress(unsigned SectionID, uint64_t Addr) {
  Dyld->reassignSectionAddress(SectionID, Addr);
}

void RuntimeDyld::mapSectionAddress(const void *LocalAddress,
                                    uint64_t TargetAddress) {
  Dyld->mapSectionAddress(LocalAddress, TargetAddress);
}

bool RuntimeDyld::hasError() { return Dyld->hasError(); }

StringRef RuntimeDyld::getErrorString() { return Dyld->getErrorString(); }

void RuntimeDyld::registerEHFrames() {
  if (Dyld)
    Dyld->registerEHFrames();
}

void RuntimeDyld::deregisterEHFrames() {
  if (Dyld)
    Dyld->deregisterEHFrames();
}

} // end namespace llvm
