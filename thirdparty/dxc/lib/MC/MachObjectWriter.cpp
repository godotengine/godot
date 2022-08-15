//===- lib/MC/MachObjectWriter.cpp - Mach-O File Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbolMachO.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "mc"

void MachObjectWriter::reset() {
  Relocations.clear();
  IndirectSymBase.clear();
  StringTable.clear();
  LocalSymbolData.clear();
  ExternalSymbolData.clear();
  UndefinedSymbolData.clear();
  MCObjectWriter::reset();
}

bool MachObjectWriter::doesSymbolRequireExternRelocation(const MCSymbol &S) {
  // Undefined symbols are always extern.
  if (S.isUndefined())
    return true;

  // References to weak definitions require external relocation entries; the
  // definition may not always be the one in the same object file.
  if (cast<MCSymbolMachO>(S).isWeakDefinition())
    return true;

  // Otherwise, we can use an internal relocation.
  return false;
}

bool MachObjectWriter::
MachSymbolData::operator<(const MachSymbolData &RHS) const {
  return Symbol->getName() < RHS.Symbol->getName();
}

bool MachObjectWriter::isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind) {
  const MCFixupKindInfo &FKI = Asm.getBackend().getFixupKindInfo(
    (MCFixupKind) Kind);

  return FKI.Flags & MCFixupKindInfo::FKF_IsPCRel;
}

uint64_t MachObjectWriter::getFragmentAddress(const MCFragment *Fragment,
                                              const MCAsmLayout &Layout) const {
  return getSectionAddress(Fragment->getParent()) +
         Layout.getFragmentOffset(Fragment);
}

uint64_t MachObjectWriter::getSymbolAddress(const MCSymbol &S,
                                            const MCAsmLayout &Layout) const {
  // If this is a variable, then recursively evaluate now.
  if (S.isVariable()) {
    if (const MCConstantExpr *C =
          dyn_cast<const MCConstantExpr>(S.getVariableValue()))
      return C->getValue();


    MCValue Target;
    if (!S.getVariableValue()->evaluateAsRelocatable(Target, &Layout, nullptr))
      report_fatal_error("unable to evaluate offset for variable '" +
                         S.getName() + "'");

    // Verify that any used symbols are defined.
    if (Target.getSymA() && Target.getSymA()->getSymbol().isUndefined())
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         Target.getSymA()->getSymbol().getName() + "'");
    if (Target.getSymB() && Target.getSymB()->getSymbol().isUndefined())
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         Target.getSymB()->getSymbol().getName() + "'");

    uint64_t Address = Target.getConstant();
    if (Target.getSymA())
      Address += getSymbolAddress(Target.getSymA()->getSymbol(), Layout);
    if (Target.getSymB())
      Address += getSymbolAddress(Target.getSymB()->getSymbol(), Layout);
    return Address;
  }

  return getSectionAddress(S.getFragment()->getParent()) +
         Layout.getSymbolOffset(S);
}

uint64_t MachObjectWriter::getPaddingSize(const MCSection *Sec,
                                          const MCAsmLayout &Layout) const {
  uint64_t EndAddr = getSectionAddress(Sec) + Layout.getSectionAddressSize(Sec);
  unsigned Next = Sec->getLayoutOrder() + 1;
  if (Next >= Layout.getSectionOrder().size())
    return 0;

  const MCSection &NextSec = *Layout.getSectionOrder()[Next];
  if (NextSec.isVirtualSection())
    return 0;
  return OffsetToAlignment(EndAddr, NextSec.getAlignment());
}

void MachObjectWriter::writeHeader(unsigned NumLoadCommands,
                                   unsigned LoadCommandsSize,
                                   bool SubsectionsViaSymbols) {
  uint32_t Flags = 0;

  if (SubsectionsViaSymbols)
    Flags |= MachO::MH_SUBSECTIONS_VIA_SYMBOLS;

  // struct mach_header (28 bytes) or
  // struct mach_header_64 (32 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  write32(is64Bit() ? MachO::MH_MAGIC_64 : MachO::MH_MAGIC);

  write32(TargetObjectWriter->getCPUType());
  write32(TargetObjectWriter->getCPUSubtype());

  write32(MachO::MH_OBJECT);
  write32(NumLoadCommands);
  write32(LoadCommandsSize);
  write32(Flags);
  if (is64Bit())
    write32(0); // reserved

  assert(OS.tell() - Start ==
         (is64Bit()?sizeof(MachO::mach_header_64): sizeof(MachO::mach_header)));
}

/// writeSegmentLoadCommand - Write a segment load command.
///
/// \param NumSections The number of sections in this segment.
/// \param SectionDataSize The total size of the sections.
void MachObjectWriter::writeSegmentLoadCommand(unsigned NumSections,
                                               uint64_t VMSize,
                                               uint64_t SectionDataStartOffset,
                                               uint64_t SectionDataSize) {
  // struct segment_command (56 bytes) or
  // struct segment_command_64 (72 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  unsigned SegmentLoadCommandSize =
    is64Bit() ? sizeof(MachO::segment_command_64):
    sizeof(MachO::segment_command);
  write32(is64Bit() ? MachO::LC_SEGMENT_64 : MachO::LC_SEGMENT);
  write32(SegmentLoadCommandSize +
          NumSections * (is64Bit() ? sizeof(MachO::section_64) :
                         sizeof(MachO::section)));

  writeBytes("", 16);
  if (is64Bit()) {
    write64(0); // vmaddr
    write64(VMSize); // vmsize
    write64(SectionDataStartOffset); // file offset
    write64(SectionDataSize); // file size
  } else {
    write32(0); // vmaddr
    write32(VMSize); // vmsize
    write32(SectionDataStartOffset); // file offset
    write32(SectionDataSize); // file size
  }
  // maxprot
  write32(MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE);
  // initprot
  write32(MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE);
  write32(NumSections);
  write32(0); // flags

  assert(OS.tell() - Start == SegmentLoadCommandSize);
}

void MachObjectWriter::writeSection(const MCAssembler &Asm,
                                    const MCAsmLayout &Layout,
                                    const MCSection &Sec, uint64_t FileOffset,
                                    uint64_t RelocationsStart,
                                    unsigned NumRelocations) {
  uint64_t SectionSize = Layout.getSectionAddressSize(&Sec);
  const MCSectionMachO &Section = cast<MCSectionMachO>(Sec);

  // The offset is unused for virtual sections.
  if (Section.isVirtualSection()) {
    assert(Layout.getSectionFileSize(&Sec) == 0 && "Invalid file size!");
    FileOffset = 0;
  }

  // struct section (68 bytes) or
  // struct section_64 (80 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  writeBytes(Section.getSectionName(), 16);
  writeBytes(Section.getSegmentName(), 16);
  if (is64Bit()) {
    write64(getSectionAddress(&Sec)); // address
    write64(SectionSize); // size
  } else {
    write32(getSectionAddress(&Sec)); // address
    write32(SectionSize); // size
  }
  write32(FileOffset);

  unsigned Flags = Section.getTypeAndAttributes();
  if (Section.hasInstructions())
    Flags |= MachO::S_ATTR_SOME_INSTRUCTIONS;

  assert(isPowerOf2_32(Section.getAlignment()) && "Invalid alignment!");
  write32(Log2_32(Section.getAlignment()));
  write32(NumRelocations ? RelocationsStart : 0);
  write32(NumRelocations);
  write32(Flags);
  write32(IndirectSymBase.lookup(&Sec)); // reserved1
  write32(Section.getStubSize()); // reserved2
  if (is64Bit())
    write32(0); // reserved3

  assert(OS.tell() - Start == (is64Bit() ? sizeof(MachO::section_64) :
                               sizeof(MachO::section)));
}

void MachObjectWriter::writeSymtabLoadCommand(uint32_t SymbolOffset,
                                              uint32_t NumSymbols,
                                              uint32_t StringTableOffset,
                                              uint32_t StringTableSize) {
  // struct symtab_command (24 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  write32(MachO::LC_SYMTAB);
  write32(sizeof(MachO::symtab_command));
  write32(SymbolOffset);
  write32(NumSymbols);
  write32(StringTableOffset);
  write32(StringTableSize);

  assert(OS.tell() - Start == sizeof(MachO::symtab_command));
}

void MachObjectWriter::writeDysymtabLoadCommand(uint32_t FirstLocalSymbol,
                                                uint32_t NumLocalSymbols,
                                                uint32_t FirstExternalSymbol,
                                                uint32_t NumExternalSymbols,
                                                uint32_t FirstUndefinedSymbol,
                                                uint32_t NumUndefinedSymbols,
                                                uint32_t IndirectSymbolOffset,
                                                uint32_t NumIndirectSymbols) {
  // struct dysymtab_command (80 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  write32(MachO::LC_DYSYMTAB);
  write32(sizeof(MachO::dysymtab_command));
  write32(FirstLocalSymbol);
  write32(NumLocalSymbols);
  write32(FirstExternalSymbol);
  write32(NumExternalSymbols);
  write32(FirstUndefinedSymbol);
  write32(NumUndefinedSymbols);
  write32(0); // tocoff
  write32(0); // ntoc
  write32(0); // modtaboff
  write32(0); // nmodtab
  write32(0); // extrefsymoff
  write32(0); // nextrefsyms
  write32(IndirectSymbolOffset);
  write32(NumIndirectSymbols);
  write32(0); // extreloff
  write32(0); // nextrel
  write32(0); // locreloff
  write32(0); // nlocrel

  assert(OS.tell() - Start == sizeof(MachO::dysymtab_command));
}

MachObjectWriter::MachSymbolData *
MachObjectWriter::findSymbolData(const MCSymbol &Sym) {
  for (auto *SymbolData :
       {&LocalSymbolData, &ExternalSymbolData, &UndefinedSymbolData})
    for (MachSymbolData &Entry : *SymbolData)
      if (Entry.Symbol == &Sym)
        return &Entry;

  return nullptr;
}

const MCSymbol &MachObjectWriter::findAliasedSymbol(const MCSymbol &Sym) const {
  const MCSymbol *S = &Sym;
  while (S->isVariable()) {
    const MCExpr *Value = S->getVariableValue();
    const auto *Ref = dyn_cast<MCSymbolRefExpr>(Value);
    if (!Ref)
      return *S;
    S = &Ref->getSymbol();
  }
  return *S;
}

void MachObjectWriter::writeNlist(MachSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  const MCSymbol *Symbol = MSD.Symbol;
  const MCSymbol &Data = *Symbol;
  const MCSymbol *AliasedSymbol = &findAliasedSymbol(*Symbol);
  uint8_t SectionIndex = MSD.SectionIndex;
  uint8_t Type = 0;
  uint64_t Address = 0;
  bool IsAlias = Symbol != AliasedSymbol;

  const MCSymbol &OrigSymbol = *Symbol;
  MachSymbolData *AliaseeInfo;
  if (IsAlias) {
    AliaseeInfo = findSymbolData(*AliasedSymbol);
    if (AliaseeInfo)
      SectionIndex = AliaseeInfo->SectionIndex;
    Symbol = AliasedSymbol;
    // FIXME: Should this update Data as well?  Do we need OrigSymbol at all?
  }

  // Set the N_TYPE bits. See <mach-o/nlist.h>.
  //
  // FIXME: Are the prebound or indirect fields possible here?
  if (IsAlias && Symbol->isUndefined())
    Type = MachO::N_INDR;
  else if (Symbol->isUndefined())
    Type = MachO::N_UNDF;
  else if (Symbol->isAbsolute())
    Type = MachO::N_ABS;
  else
    Type = MachO::N_SECT;

  // FIXME: Set STAB bits.

  if (Data.isPrivateExtern())
    Type |= MachO::N_PEXT;

  // Set external bit.
  if (Data.isExternal() || (!IsAlias && Symbol->isUndefined()))
    Type |= MachO::N_EXT;

  // Compute the symbol address.
  if (IsAlias && Symbol->isUndefined())
    Address = AliaseeInfo->StringIndex;
  else if (Symbol->isDefined())
    Address = getSymbolAddress(OrigSymbol, Layout);
  else if (Symbol->isCommon()) {
    // Common symbols are encoded with the size in the address
    // field, and their alignment in the flags.
    Address = Symbol->getCommonSize();
  }

  // struct nlist (12 bytes)

  write32(MSD.StringIndex);
  write8(Type);
  write8(SectionIndex);

  // The Mach-O streamer uses the lowest 16-bits of the flags for the 'desc'
  // value.
  write16(cast<MCSymbolMachO>(Symbol)->getEncodedFlags());
  if (is64Bit())
    write64(Address);
  else
    write32(Address);
}

void MachObjectWriter::writeLinkeditLoadCommand(uint32_t Type,
                                                uint32_t DataOffset,
                                                uint32_t DataSize) {
  uint64_t Start = OS.tell();
  (void) Start;

  write32(Type);
  write32(sizeof(MachO::linkedit_data_command));
  write32(DataOffset);
  write32(DataSize);

  assert(OS.tell() - Start == sizeof(MachO::linkedit_data_command));
}

static unsigned ComputeLinkerOptionsLoadCommandSize(
  const std::vector<std::string> &Options, bool is64Bit)
{
  unsigned Size = sizeof(MachO::linker_option_command);
  for (const std::string &Option : Options)
    Size += Option.size() + 1;
  return RoundUpToAlignment(Size, is64Bit ? 8 : 4);
}

void MachObjectWriter::writeLinkerOptionsLoadCommand(
  const std::vector<std::string> &Options)
{
  unsigned Size = ComputeLinkerOptionsLoadCommandSize(Options, is64Bit());
  uint64_t Start = OS.tell();
  (void) Start;

  write32(MachO::LC_LINKER_OPTION);
  write32(Size);
  write32(Options.size());
  uint64_t BytesWritten = sizeof(MachO::linker_option_command);
  for (const std::string &Option : Options) {
    // Write each string, including the null byte.
    writeBytes(Option.c_str(), Option.size() + 1);
    BytesWritten += Option.size() + 1;
  }

  // Pad to a multiple of the pointer size.
  writeBytes("", OffsetToAlignment(BytesWritten, is64Bit() ? 8 : 4));

  assert(OS.tell() - Start == Size);
}

void MachObjectWriter::recordRelocation(MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCFixup &Fixup, MCValue Target,
                                        bool &IsPCRel, uint64_t &FixedValue) {
  TargetObjectWriter->recordRelocation(this, Asm, Layout, Fragment, Fixup,
                                       Target, FixedValue);
}

void MachObjectWriter::bindIndirectSymbols(MCAssembler &Asm) {
  // This is the point where 'as' creates actual symbols for indirect symbols
  // (in the following two passes). It would be easier for us to do this sooner
  // when we see the attribute, but that makes getting the order in the symbol
  // table much more complicated than it is worth.
  //
  // FIXME: Revisit this when the dust settles.

  // Report errors for use of .indirect_symbol not in a symbol pointer section
  // or stub section.
  for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
         ie = Asm.indirect_symbol_end(); it != ie; ++it) {
    const MCSectionMachO &Section = cast<MCSectionMachO>(*it->Section);

    if (Section.getType() != MachO::S_NON_LAZY_SYMBOL_POINTERS &&
        Section.getType() != MachO::S_LAZY_SYMBOL_POINTERS &&
        Section.getType() != MachO::S_SYMBOL_STUBS) {
	MCSymbol &Symbol = *it->Symbol;
	report_fatal_error("indirect symbol '" + Symbol.getName() +
                           "' not in a symbol pointer or stub section");
    }
  }

  // Bind non-lazy symbol pointers first.
  unsigned IndirectIndex = 0;
  for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
         ie = Asm.indirect_symbol_end(); it != ie; ++it, ++IndirectIndex) {
    const MCSectionMachO &Section = cast<MCSectionMachO>(*it->Section);

    if (Section.getType() != MachO::S_NON_LAZY_SYMBOL_POINTERS)
      continue;

    // Initialize the section indirect symbol base, if necessary.
    IndirectSymBase.insert(std::make_pair(it->Section, IndirectIndex));

    Asm.registerSymbol(*it->Symbol);
  }

  // Then lazy symbol pointers and symbol stubs.
  IndirectIndex = 0;
  for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
         ie = Asm.indirect_symbol_end(); it != ie; ++it, ++IndirectIndex) {
    const MCSectionMachO &Section = cast<MCSectionMachO>(*it->Section);

    if (Section.getType() != MachO::S_LAZY_SYMBOL_POINTERS &&
        Section.getType() != MachO::S_SYMBOL_STUBS)
      continue;

    // Initialize the section indirect symbol base, if necessary.
    IndirectSymBase.insert(std::make_pair(it->Section, IndirectIndex));

    // Set the symbol type to undefined lazy, but only on construction.
    //
    // FIXME: Do not hardcode.
    bool Created;
    Asm.registerSymbol(*it->Symbol, &Created);
    if (Created)
      cast<MCSymbolMachO>(it->Symbol)->setReferenceTypeUndefinedLazy(true);
  }
}

/// computeSymbolTable - Compute the symbol table data
void MachObjectWriter::computeSymbolTable(
    MCAssembler &Asm, std::vector<MachSymbolData> &LocalSymbolData,
    std::vector<MachSymbolData> &ExternalSymbolData,
    std::vector<MachSymbolData> &UndefinedSymbolData) {
  // Build section lookup table.
  DenseMap<const MCSection*, uint8_t> SectionIndexMap;
  unsigned Index = 1;
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it, ++Index)
    SectionIndexMap[&*it] = Index;
  assert(Index <= 256 && "Too many sections!");

  // Build the string table.
  for (const MCSymbol &Symbol : Asm.symbols()) {
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    StringTable.add(Symbol.getName());
  }
  StringTable.finalize(StringTableBuilder::MachO);

  // Build the symbol arrays but only for non-local symbols.
  //
  // The particular order that we collect and then sort the symbols is chosen to
  // match 'as'. Even though it doesn't matter for correctness, this is
  // important for letting us diff .o files.
  for (const MCSymbol &Symbol : Asm.symbols()) {
    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (!Symbol.isExternal() && !Symbol.isUndefined())
      continue;

    MachSymbolData MSD;
    MSD.Symbol = &Symbol;
    MSD.StringIndex = StringTable.getOffset(Symbol.getName());

    if (Symbol.isUndefined()) {
      MSD.SectionIndex = 0;
      UndefinedSymbolData.push_back(MSD);
    } else if (Symbol.isAbsolute()) {
      MSD.SectionIndex = 0;
      ExternalSymbolData.push_back(MSD);
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      ExternalSymbolData.push_back(MSD);
    }
  }

  // Now add the data for local symbols.
  for (const MCSymbol &Symbol : Asm.symbols()) {
    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (Symbol.isExternal() || Symbol.isUndefined())
      continue;

    MachSymbolData MSD;
    MSD.Symbol = &Symbol;
    MSD.StringIndex = StringTable.getOffset(Symbol.getName());

    if (Symbol.isAbsolute()) {
      MSD.SectionIndex = 0;
      LocalSymbolData.push_back(MSD);
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      LocalSymbolData.push_back(MSD);
    }
  }

  // External and undefined symbols are required to be in lexicographic order.
  std::sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  std::sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices.
  Index = 0;
  for (auto *SymbolData :
       {&LocalSymbolData, &ExternalSymbolData, &UndefinedSymbolData})
    for (MachSymbolData &Entry : *SymbolData)
      Entry.Symbol->setIndex(Index++);

  for (const MCSection &Section : Asm) {
    for (RelAndSymbol &Rel : Relocations[&Section]) {
      if (!Rel.Sym)
        continue;

      // Set the Index and the IsExtern bit.
      unsigned Index = Rel.Sym->getIndex();
      assert(isInt<24>(Index));
      if (IsLittleEndian)
        Rel.MRE.r_word1 = (Rel.MRE.r_word1 & (~0U << 24)) | Index | (1 << 27);
      else
        Rel.MRE.r_word1 = (Rel.MRE.r_word1 & 0xff) | Index << 8 | (1 << 4);
    }
  }
}

void MachObjectWriter::computeSectionAddresses(const MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  uint64_t StartAddress = 0;
  for (const MCSection *Sec : Layout.getSectionOrder()) {
    StartAddress = RoundUpToAlignment(StartAddress, Sec->getAlignment());
    SectionAddress[Sec] = StartAddress;
    StartAddress += Layout.getSectionAddressSize(Sec);

    // Explicitly pad the section to match the alignment requirements of the
    // following one. This is for 'gas' compatibility, it shouldn't
    /// strictly be necessary.
    StartAddress += getPaddingSize(Sec, Layout);
  }
}

void MachObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
  computeSectionAddresses(Asm, Layout);

  // Create symbol data for any indirect symbols.
  bindIndirectSymbols(Asm);
}

bool MachObjectWriter::isSymbolRefDifferenceFullyResolvedImpl(
    const MCAssembler &Asm, const MCSymbol &SymA, const MCFragment &FB,
    bool InSet, bool IsPCRel) const {
  if (InSet)
    return true;

  // The effective address is
  //     addr(atom(A)) + offset(A)
  //   - addr(atom(B)) - offset(B)
  // and the offsets are not relocatable, so the fixup is fully resolved when
  //  addr(atom(A)) - addr(atom(B)) == 0.
  const MCSymbol &SA = findAliasedSymbol(SymA);
  const MCSection &SecA = SA.getSection();
  const MCSection &SecB = *FB.getParent();

  if (IsPCRel) {
    // The simple (Darwin, except on x86_64) way of dealing with this was to
    // assume that any reference to a temporary symbol *must* be a temporary
    // symbol in the same atom, unless the sections differ. Therefore, any PCrel
    // relocation to a temporary symbol (in the same section) is fully
    // resolved. This also works in conjunction with absolutized .set, which
    // requires the compiler to use .set to absolutize the differences between
    // symbols which the compiler knows to be assembly time constants, so we
    // don't need to worry about considering symbol differences fully resolved.
    //
    // If the file isn't using sub-sections-via-symbols, we can make the
    // same assumptions about any symbol that we normally make about
    // assembler locals.

    bool hasReliableSymbolDifference = isX86_64();
    if (!hasReliableSymbolDifference) {
      if (!SA.isInSection() || &SecA != &SecB ||
          (!SA.isTemporary() && FB.getAtom() != SA.getFragment()->getAtom() &&
           Asm.getSubsectionsViaSymbols()))
        return false;
      return true;
    }
    // For Darwin x86_64, there is one special case when the reference IsPCRel.
    // If the fragment with the reference does not have a base symbol but meets
    // the simple way of dealing with this, in that it is a temporary symbol in
    // the same atom then it is assumed to be fully resolved.  This is needed so
    // a relocation entry is not created and so the static linker does not
    // mess up the reference later.
    else if(!FB.getAtom() &&
            SA.isTemporary() && SA.isInSection() && &SecA == &SecB){
      return true;
    }
  }

  // If they are not in the same section, we can't compute the diff.
  if (&SecA != &SecB)
    return false;

  const MCFragment *FA = SA.getFragment();

  // Bail if the symbol has no fragment.
  if (!FA)
    return false;

  // If the atoms are the same, they are guaranteed to have the same address.
  if (FA->getAtom() == FB.getAtom())
    return true;

  // Otherwise, we can't prove this is fully resolved.
  return false;
}

void MachObjectWriter::writeObject(MCAssembler &Asm,
                                   const MCAsmLayout &Layout) {
  // Compute symbol table information and bind symbol indices.
  computeSymbolTable(Asm, LocalSymbolData, ExternalSymbolData,
                     UndefinedSymbolData);

  unsigned NumSections = Asm.size();
  const MCAssembler::VersionMinInfoType &VersionInfo =
    Layout.getAssembler().getVersionMinInfo();

  // The section data starts after the header, the segment load command (and
  // section headers) and the symbol table.
  unsigned NumLoadCommands = 1;
  uint64_t LoadCommandsSize = is64Bit() ?
    sizeof(MachO::segment_command_64) + NumSections * sizeof(MachO::section_64):
    sizeof(MachO::segment_command) + NumSections * sizeof(MachO::section);

  // Add the deployment target version info load command size, if used.
  if (VersionInfo.Major != 0) {
    ++NumLoadCommands;
    LoadCommandsSize += sizeof(MachO::version_min_command);
  }

  // Add the data-in-code load command size, if used.
  unsigned NumDataRegions = Asm.getDataRegions().size();
  if (NumDataRegions) {
    ++NumLoadCommands;
    LoadCommandsSize += sizeof(MachO::linkedit_data_command);
  }

  // Add the loh load command size, if used.
  uint64_t LOHRawSize = Asm.getLOHContainer().getEmitSize(*this, Layout);
  uint64_t LOHSize = RoundUpToAlignment(LOHRawSize, is64Bit() ? 8 : 4);
  if (LOHSize) {
    ++NumLoadCommands;
    LoadCommandsSize += sizeof(MachO::linkedit_data_command);
  }

  // Add the symbol table load command sizes, if used.
  unsigned NumSymbols = LocalSymbolData.size() + ExternalSymbolData.size() +
    UndefinedSymbolData.size();
  if (NumSymbols) {
    NumLoadCommands += 2;
    LoadCommandsSize += (sizeof(MachO::symtab_command) +
                         sizeof(MachO::dysymtab_command));
  }

  // Add the linker option load commands sizes.
  for (const auto &Option : Asm.getLinkerOptions()) {
    ++NumLoadCommands;
    LoadCommandsSize += ComputeLinkerOptionsLoadCommandSize(Option, is64Bit());
  }
  
  // Compute the total size of the section data, as well as its file size and vm
  // size.
  uint64_t SectionDataStart = (is64Bit() ? sizeof(MachO::mach_header_64) :
                               sizeof(MachO::mach_header)) + LoadCommandsSize;
  uint64_t SectionDataSize = 0;
  uint64_t SectionDataFileSize = 0;
  uint64_t VMSize = 0;
  for (const MCSection &Sec : Asm) {
    uint64_t Address = getSectionAddress(&Sec);
    uint64_t Size = Layout.getSectionAddressSize(&Sec);
    uint64_t FileSize = Layout.getSectionFileSize(&Sec);
    FileSize += getPaddingSize(&Sec, Layout);

    VMSize = std::max(VMSize, Address + Size);

    if (Sec.isVirtualSection())
      continue;

    SectionDataSize = std::max(SectionDataSize, Address + Size);
    SectionDataFileSize = std::max(SectionDataFileSize, Address + FileSize);
  }

  // The section data is padded to 4 bytes.
  //
  // FIXME: Is this machine dependent?
  unsigned SectionDataPadding = OffsetToAlignment(SectionDataFileSize, 4);
  SectionDataFileSize += SectionDataPadding;

  // Write the prolog, starting with the header and load command...
  writeHeader(NumLoadCommands, LoadCommandsSize,
              Asm.getSubsectionsViaSymbols());
  writeSegmentLoadCommand(NumSections, VMSize,
                          SectionDataStart, SectionDataSize);

  // ... and then the section headers.
  uint64_t RelocTableEnd = SectionDataStart + SectionDataFileSize;
  for (const MCSection &Sec : Asm) {
    std::vector<RelAndSymbol> &Relocs = Relocations[&Sec];
    unsigned NumRelocs = Relocs.size();
    uint64_t SectionStart = SectionDataStart + getSectionAddress(&Sec);
    writeSection(Asm, Layout, Sec, SectionStart, RelocTableEnd, NumRelocs);
    RelocTableEnd += NumRelocs * sizeof(MachO::any_relocation_info);
  }

  // Write out the deployment target information, if it's available.
  if (VersionInfo.Major != 0) {
    assert(VersionInfo.Update < 256 && "unencodable update target version");
    assert(VersionInfo.Minor < 256 && "unencodable minor target version");
    assert(VersionInfo.Major < 65536 && "unencodable major target version");
    uint32_t EncodedVersion = VersionInfo.Update | (VersionInfo.Minor << 8) |
      (VersionInfo.Major << 16);
    write32(VersionInfo.Kind == MCVM_OSXVersionMin ? MachO::LC_VERSION_MIN_MACOSX :
            MachO::LC_VERSION_MIN_IPHONEOS);
    write32(sizeof(MachO::version_min_command));
    write32(EncodedVersion);
    write32(0);         // reserved.
  }

  // Write the data-in-code load command, if used.
  uint64_t DataInCodeTableEnd = RelocTableEnd + NumDataRegions * 8;
  if (NumDataRegions) {
    uint64_t DataRegionsOffset = RelocTableEnd;
    uint64_t DataRegionsSize = NumDataRegions * 8;
    writeLinkeditLoadCommand(MachO::LC_DATA_IN_CODE, DataRegionsOffset,
                             DataRegionsSize);
  }

  // Write the loh load command, if used.
  uint64_t LOHTableEnd = DataInCodeTableEnd + LOHSize;
  if (LOHSize)
    writeLinkeditLoadCommand(MachO::LC_LINKER_OPTIMIZATION_HINT,
                             DataInCodeTableEnd, LOHSize);

  // Write the symbol table load command, if used.
  if (NumSymbols) {
    unsigned FirstLocalSymbol = 0;
    unsigned NumLocalSymbols = LocalSymbolData.size();
    unsigned FirstExternalSymbol = FirstLocalSymbol + NumLocalSymbols;
    unsigned NumExternalSymbols = ExternalSymbolData.size();
    unsigned FirstUndefinedSymbol = FirstExternalSymbol + NumExternalSymbols;
    unsigned NumUndefinedSymbols = UndefinedSymbolData.size();
    unsigned NumIndirectSymbols = Asm.indirect_symbol_size();
    unsigned NumSymTabSymbols =
      NumLocalSymbols + NumExternalSymbols + NumUndefinedSymbols;
    uint64_t IndirectSymbolSize = NumIndirectSymbols * 4;
    uint64_t IndirectSymbolOffset = 0;

    // If used, the indirect symbols are written after the section data.
    if (NumIndirectSymbols)
      IndirectSymbolOffset = LOHTableEnd;

    // The symbol table is written after the indirect symbol data.
    uint64_t SymbolTableOffset = LOHTableEnd + IndirectSymbolSize;

    // The string table is written after symbol table.
    uint64_t StringTableOffset =
      SymbolTableOffset + NumSymTabSymbols * (is64Bit() ?
                                              sizeof(MachO::nlist_64) :
                                              sizeof(MachO::nlist));
    writeSymtabLoadCommand(SymbolTableOffset, NumSymTabSymbols,
                           StringTableOffset, StringTable.data().size());

    writeDysymtabLoadCommand(FirstLocalSymbol, NumLocalSymbols,
                             FirstExternalSymbol, NumExternalSymbols,
                             FirstUndefinedSymbol, NumUndefinedSymbols,
                             IndirectSymbolOffset, NumIndirectSymbols);
  }

  // Write the linker options load commands.
  for (const auto &Option : Asm.getLinkerOptions())
    writeLinkerOptionsLoadCommand(Option);

  // Write the actual section data.
  for (const MCSection &Sec : Asm) {
    Asm.writeSectionData(&Sec, Layout);

    uint64_t Pad = getPaddingSize(&Sec, Layout);
    WriteZeros(Pad);
  }

  // Write the extra padding.
  WriteZeros(SectionDataPadding);

  // Write the relocation entries.
  for (const MCSection &Sec : Asm) {
    // Write the section relocation entries, in reverse order to match 'as'
    // (approximately, the exact algorithm is more complicated than this).
    std::vector<RelAndSymbol> &Relocs = Relocations[&Sec];
    for (const RelAndSymbol &Rel : make_range(Relocs.rbegin(), Relocs.rend())) {
      write32(Rel.MRE.r_word0);
      write32(Rel.MRE.r_word1);
    }
  }

  // Write out the data-in-code region payload, if there is one.
  for (MCAssembler::const_data_region_iterator
         it = Asm.data_region_begin(), ie = Asm.data_region_end();
         it != ie; ++it) {
    const DataRegionData *Data = &(*it);
    uint64_t Start = getSymbolAddress(*Data->Start, Layout);
    uint64_t End = getSymbolAddress(*Data->End, Layout);
    DEBUG(dbgs() << "data in code region-- kind: " << Data->Kind
                 << "  start: " << Start << "(" << Data->Start->getName() << ")"
                 << "  end: " << End << "(" << Data->End->getName() << ")"
                 << "  size: " << End - Start
                 << "\n");
    write32(Start);
    write16(End - Start);
    write16(Data->Kind);
  }

  // Write out the loh commands, if there is one.
  if (LOHSize) {
#ifndef NDEBUG
    unsigned Start = OS.tell();
#endif
    Asm.getLOHContainer().emit(*this, Layout);
    // Pad to a multiple of the pointer size.
    writeBytes("", OffsetToAlignment(LOHRawSize, is64Bit() ? 8 : 4));
    assert(OS.tell() - Start == LOHSize);
  }

  // Write the symbol table data, if used.
  if (NumSymbols) {
    // Write the indirect symbol entries.
    for (MCAssembler::const_indirect_symbol_iterator
           it = Asm.indirect_symbol_begin(),
           ie = Asm.indirect_symbol_end(); it != ie; ++it) {
      // Indirect symbols in the non-lazy symbol pointer section have some
      // special handling.
      const MCSectionMachO &Section =
          static_cast<const MCSectionMachO &>(*it->Section);
      if (Section.getType() == MachO::S_NON_LAZY_SYMBOL_POINTERS) {
        // If this symbol is defined and internal, mark it as such.
        if (it->Symbol->isDefined() && !it->Symbol->isExternal()) {
          uint32_t Flags = MachO::INDIRECT_SYMBOL_LOCAL;
          if (it->Symbol->isAbsolute())
            Flags |= MachO::INDIRECT_SYMBOL_ABS;
          write32(Flags);
          continue;
        }
      }

      write32(it->Symbol->getIndex());
    }

    // FIXME: Check that offsets match computed ones.

    // Write the symbol table entries.
    for (auto *SymbolData :
         {&LocalSymbolData, &ExternalSymbolData, &UndefinedSymbolData})
      for (MachSymbolData &Entry : *SymbolData)
        writeNlist(Entry, Layout);

    // Write the string table.
    OS << StringTable.data();
  }
}

MCObjectWriter *llvm::createMachObjectWriter(MCMachObjectTargetWriter *MOTW,
                                             raw_pwrite_stream &OS,
                                             bool IsLittleEndian) {
  return new MachObjectWriter(MOTW, OS, IsLittleEndian);
}
