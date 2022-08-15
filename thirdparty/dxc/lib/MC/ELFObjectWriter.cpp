//===- lib/MC/ELFObjectWriter.cpp - ELF File Writer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>
using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

namespace {

typedef DenseMap<const MCSectionELF *, uint32_t> SectionIndexMapTy;

class ELFObjectWriter;

class SymbolTableWriter {
  ELFObjectWriter &EWriter;
  bool Is64Bit;

  // indexes we are going to write to .symtab_shndx.
  std::vector<uint32_t> ShndxIndexes;

  // The numbel of symbols written so far.
  unsigned NumWritten;

  void createSymtabShndx();

  template <typename T> void write(T Value);

public:
  SymbolTableWriter(ELFObjectWriter &EWriter, bool Is64Bit);

  void writeSymbol(uint32_t name, uint8_t info, uint64_t value, uint64_t size,
                   uint8_t other, uint32_t shndx, bool Reserved);

  ArrayRef<uint32_t> getShndxIndexes() const { return ShndxIndexes; }
};

class ELFObjectWriter : public MCObjectWriter {
    static bool isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind);
    static uint64_t SymbolValue(const MCSymbol &Sym, const MCAsmLayout &Layout);
    static bool isInSymtab(const MCAsmLayout &Layout, const MCSymbolELF &Symbol,
                           bool Used, bool Renamed);

    /// Helper struct for containing some precomputed information on symbols.
    struct ELFSymbolData {
      const MCSymbolELF *Symbol;
      uint32_t SectionIndex;
      StringRef Name;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        unsigned LHSType = Symbol->getType();
        unsigned RHSType = RHS.Symbol->getType();
        if (LHSType == ELF::STT_SECTION && RHSType != ELF::STT_SECTION)
          return false;
        if (LHSType != ELF::STT_SECTION && RHSType == ELF::STT_SECTION)
          return true;
        if (LHSType == ELF::STT_SECTION && RHSType == ELF::STT_SECTION)
          return SectionIndex < RHS.SectionIndex;
        return Name < RHS.Name;
      }
    };

    /// The target specific ELF writer instance.
    std::unique_ptr<MCELFObjectTargetWriter> TargetObjectWriter;

    DenseMap<const MCSymbolELF *, const MCSymbolELF *> Renames;

    llvm::DenseMap<const MCSectionELF *, std::vector<ELFRelocationEntry>>
        Relocations;

    /// @}
    /// @name Symbol Table Data
    /// @{

    StringTableBuilder StrTabBuilder;

    /// @}

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;
    // This holds the .symtab section index.
    unsigned SymbolTableIndex;

    // Sections in the order they are to be output in the section table.
    std::vector<const MCSectionELF *> SectionTable;
    unsigned addToSectionTable(const MCSectionELF *Sec);

    // TargetObjectWriter wrappers.
    bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
    bool hasRelocationAddend() const {
      return TargetObjectWriter->hasRelocationAddend();
    }
    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel) const {
      return TargetObjectWriter->GetRelocType(Target, Fixup, IsPCRel);
    }

    void align(unsigned Alignment);

  public:
    ELFObjectWriter(MCELFObjectTargetWriter *MOTW, raw_pwrite_stream &OS,
                    bool IsLittleEndian)
        : MCObjectWriter(OS, IsLittleEndian), TargetObjectWriter(MOTW) {}

    void reset() override {
      Renames.clear();
      Relocations.clear();
      StrTabBuilder.clear();
      SectionTable.clear();
      MCObjectWriter::reset();
    }

    ~ELFObjectWriter() override;

    void WriteWord(uint64_t W) {
      if (is64Bit())
        write64(W);
      else
        write32(W);
    }

    template <typename T> void write(T Val) {
      if (IsLittleEndian)
        support::endian::Writer<support::little>(OS).write(Val);
      else
        support::endian::Writer<support::big>(OS).write(Val);
    }

    void writeHeader(const MCAssembler &Asm);

    void writeSymbol(SymbolTableWriter &Writer, uint32_t StringIndex,
                     ELFSymbolData &MSD, const MCAsmLayout &Layout);

    // Start and end offset of each section
    typedef std::map<const MCSectionELF *, std::pair<uint64_t, uint64_t>>
        SectionOffsetsTy;

    bool shouldRelocateWithSymbol(const MCAssembler &Asm,
                                  const MCSymbolRefExpr *RefA,
                                  const MCSymbol *Sym, uint64_t C,
                                  unsigned Type) const;

    void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment *Fragment, const MCFixup &Fixup,
                          MCValue Target, bool &IsPCRel,
                          uint64_t &FixedValue) override;

    // Map from a signature symbol to the group section index
    typedef DenseMap<const MCSymbol *, unsigned> RevGroupMapTy;

    /// Compute the symbol table data
    ///
    /// \param Asm - The assembler.
    /// \param SectionIndexMap - Maps a section to its index.
    /// \param RevGroupMap - Maps a signature symbol to the group section.
    void computeSymbolTable(MCAssembler &Asm, const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const RevGroupMapTy &RevGroupMap,
                            SectionOffsetsTy &SectionOffsets);

    MCSectionELF *createRelocationSection(MCContext &Ctx,
                                          const MCSectionELF &Sec);

    const MCSectionELF *createStringTable(MCContext &Ctx);

    void executePostLayoutBinding(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) override;

    void writeSectionHeader(const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const SectionOffsetsTy &SectionOffsets);

    void writeSectionData(const MCAssembler &Asm, MCSection &Sec,
                          const MCAsmLayout &Layout);

    void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset, uint64_t Size,
                          uint32_t Link, uint32_t Info, uint64_t Alignment,
                          uint64_t EntrySize);

    void writeRelocations(const MCAssembler &Asm, const MCSectionELF &Sec);

    bool isSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                                const MCSymbol &SymA,
                                                const MCFragment &FB,
                                                bool InSet,
                                                bool IsPCRel) const override;

    bool isWeak(const MCSymbol &Sym) const override;

    void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
    void writeSection(const SectionIndexMapTy &SectionIndexMap,
                      uint32_t GroupSymbolIndex, uint64_t Offset, uint64_t Size,
                      const MCSectionELF &Section);
  };
}

void ELFObjectWriter::align(unsigned Alignment) {
  uint64_t Padding = OffsetToAlignment(OS.tell(), Alignment);
  WriteZeros(Padding);
}

unsigned ELFObjectWriter::addToSectionTable(const MCSectionELF *Sec) {
  SectionTable.push_back(Sec);
  StrTabBuilder.add(Sec->getSectionName());
  return SectionTable.size();
}

void SymbolTableWriter::createSymtabShndx() {
  if (!ShndxIndexes.empty())
    return;

  ShndxIndexes.resize(NumWritten);
}

template <typename T> void SymbolTableWriter::write(T Value) {
  EWriter.write(Value);
}

SymbolTableWriter::SymbolTableWriter(ELFObjectWriter &EWriter, bool Is64Bit)
    : EWriter(EWriter), Is64Bit(Is64Bit), NumWritten(0) {}

void SymbolTableWriter::writeSymbol(uint32_t name, uint8_t info, uint64_t value,
                                    uint64_t size, uint8_t other,
                                    uint32_t shndx, bool Reserved) {
  bool LargeIndex = shndx >= ELF::SHN_LORESERVE && !Reserved;

  if (LargeIndex)
    createSymtabShndx();

  if (!ShndxIndexes.empty()) {
    if (LargeIndex)
      ShndxIndexes.push_back(shndx);
    else
      ShndxIndexes.push_back(0);
  }

  uint16_t Index = LargeIndex ? uint16_t(ELF::SHN_XINDEX) : shndx;

  if (Is64Bit) {
    write(name);  // st_name
    write(info);  // st_info
    write(other); // st_other
    write(Index); // st_shndx
    write(value); // st_value
    write(size);  // st_size
  } else {
    write(name);            // st_name
    write(uint32_t(value)); // st_value
    write(uint32_t(size));  // st_size
    write(info);            // st_info
    write(other);           // st_other
    write(Index);           // st_shndx
  }

  ++NumWritten;
}

bool ELFObjectWriter::isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind) {
  const MCFixupKindInfo &FKI =
    Asm.getBackend().getFixupKindInfo((MCFixupKind) Kind);

  return FKI.Flags & MCFixupKindInfo::FKF_IsPCRel;
}

ELFObjectWriter::~ELFObjectWriter()
{}

// Emit the ELF header.
void ELFObjectWriter::writeHeader(const MCAssembler &Asm) {
  // ELF Header
  // ----------
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the latter.

  writeBytes(ELF::ElfMagic); // e_ident[EI_MAG0] to e_ident[EI_MAG3]

  write8(is64Bit() ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  write8(isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  // e_ident[EI_OSABI]
  write8(TargetObjectWriter->getOSABI());
  write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  write16(ELF::ET_REL);             // e_type

  write16(TargetObjectWriter->getEMachine()); // e_machine = target

  write32(ELF::EV_CURRENT);         // e_version
  WriteWord(0);                    // e_entry, no entry point in .o file
  WriteWord(0);                    // e_phoff, no program header for .o
  WriteWord(0);                     // e_shoff = sec hdr table off in bytes

  // e_flags = whatever the target wants
  write32(Asm.getELFHeaderEFlags());

  // e_ehsize = ELF header size
  write16(is64Bit() ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr));

  write16(0);                  // e_phentsize = prog header entry size
  write16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  write16(is64Bit() ? sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr));

  // e_shnum     = # of section header ents
  write16(0);

  // e_shstrndx  = Section # of '.shstrtab'
  assert(StringTableIndex < ELF::SHN_LORESERVE);
  write16(StringTableIndex);
}

uint64_t ELFObjectWriter::SymbolValue(const MCSymbol &Sym,
                                      const MCAsmLayout &Layout) {
  if (Sym.isCommon() && Sym.isExternal())
    return Sym.getCommonAlignment();

  uint64_t Res;
  if (!Layout.getSymbolOffset(Sym, Res))
    return 0;

  if (Layout.getAssembler().isThumbFunc(&Sym))
    Res |= 1;

  return Res;
}

void ELFObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  // The presence of symbol versions causes undefined symbols and
  // versions declared with @@@ to be renamed.

  for (const MCSymbol &A : Asm.symbols()) {
    const auto &Alias = cast<MCSymbolELF>(A);
    // Not an alias.
    if (!Alias.isVariable())
      continue;
    auto *Ref = dyn_cast<MCSymbolRefExpr>(Alias.getVariableValue());
    if (!Ref)
      continue;
    const auto &Symbol = cast<MCSymbolELF>(Ref->getSymbol());

    StringRef AliasName = Alias.getName();
    size_t Pos = AliasName.find('@');
    if (Pos == StringRef::npos)
      continue;

    // Aliases defined with .symvar copy the binding from the symbol they alias.
    // This is the first place we are able to copy this information.
    Alias.setExternal(Symbol.isExternal());
    Alias.setBinding(Symbol.getBinding());

    StringRef Rest = AliasName.substr(Pos);
    if (!Symbol.isUndefined() && !Rest.startswith("@@@"))
      continue;

    // FIXME: produce a better error message.
    if (Symbol.isUndefined() && Rest.startswith("@@") &&
        !Rest.startswith("@@@"))
      report_fatal_error("A @@ version cannot be undefined");

    Renames.insert(std::make_pair(&Symbol, &Alias));
  }
}

static uint8_t mergeTypeForSet(uint8_t origType, uint8_t newType) {
  uint8_t Type = newType;

  // Propagation rules:
  // IFUNC > FUNC > OBJECT > NOTYPE
  // TLS_OBJECT > OBJECT > NOTYPE
  //
  // dont let the new type degrade the old type
  switch (origType) {
  default:
    break;
  case ELF::STT_GNU_IFUNC:
    if (Type == ELF::STT_FUNC || Type == ELF::STT_OBJECT ||
        Type == ELF::STT_NOTYPE || Type == ELF::STT_TLS)
      Type = ELF::STT_GNU_IFUNC;
    break;
  case ELF::STT_FUNC:
    if (Type == ELF::STT_OBJECT || Type == ELF::STT_NOTYPE ||
        Type == ELF::STT_TLS)
      Type = ELF::STT_FUNC;
    break;
  case ELF::STT_OBJECT:
    if (Type == ELF::STT_NOTYPE)
      Type = ELF::STT_OBJECT;
    break;
  case ELF::STT_TLS:
    if (Type == ELF::STT_OBJECT || Type == ELF::STT_NOTYPE ||
        Type == ELF::STT_GNU_IFUNC || Type == ELF::STT_FUNC)
      Type = ELF::STT_TLS;
    break;
  }

  return Type;
}

void ELFObjectWriter::writeSymbol(SymbolTableWriter &Writer,
                                  uint32_t StringIndex, ELFSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  const auto &Symbol = cast<MCSymbolELF>(*MSD.Symbol);
  assert((!Symbol.getFragment() ||
          (Symbol.getFragment()->getParent() == &Symbol.getSection())) &&
         "The symbol's section doesn't match the fragment's symbol");
  const MCSymbolELF *Base =
      cast_or_null<MCSymbolELF>(Layout.getBaseSymbol(Symbol));

  // This has to be in sync with when computeSymbolTable uses SHN_ABS or
  // SHN_COMMON.
  bool IsReserved = !Base || Symbol.isCommon();

  // Binding and Type share the same byte as upper and lower nibbles
  uint8_t Binding = Symbol.getBinding();
  uint8_t Type = Symbol.getType();
  if (Base) {
    Type = mergeTypeForSet(Type, Base->getType());
  }
  uint8_t Info = (Binding << 4) | Type;

  // Other and Visibility share the same byte with Visibility using the lower
  // 2 bits
  uint8_t Visibility = Symbol.getVisibility();
  uint8_t Other = Symbol.getOther() | Visibility;

  uint64_t Value = SymbolValue(*MSD.Symbol, Layout);
  uint64_t Size = 0;

  const MCExpr *ESize = MSD.Symbol->getSize();
  if (!ESize && Base)
    ESize = Base->getSize();

  if (ESize) {
    int64_t Res;
    if (!ESize->evaluateKnownAbsolute(Res, Layout))
      report_fatal_error("Size expression must be absolute.");
    Size = Res;
  }

  // Write out the symbol table entry
  Writer.writeSymbol(StringIndex, Info, Value, Size, Other, MSD.SectionIndex,
                     IsReserved);
}

// It is always valid to create a relocation with a symbol. It is preferable
// to use a relocation with a section if that is possible. Using the section
// allows us to omit some local symbols from the symbol table.
bool ELFObjectWriter::shouldRelocateWithSymbol(const MCAssembler &Asm,
                                               const MCSymbolRefExpr *RefA,
                                               const MCSymbol *S, uint64_t C,
                                               unsigned Type) const {
  const auto *Sym = cast_or_null<MCSymbolELF>(S);
  // A PCRel relocation to an absolute value has no symbol (or section). We
  // represent that with a relocation to a null section.
  if (!RefA)
    return false;

  MCSymbolRefExpr::VariantKind Kind = RefA->getKind();
  switch (Kind) {
  default:
    break;
  // The .odp creation emits a relocation against the symbol ".TOC." which
  // create a R_PPC64_TOC relocation. However the relocation symbol name
  // in final object creation should be NULL, since the symbol does not
  // really exist, it is just the reference to TOC base for the current
  // object file. Since the symbol is undefined, returning false results
  // in a relocation with a null section which is the desired result.
  case MCSymbolRefExpr::VK_PPC_TOCBASE:
    return false;

  // These VariantKind cause the relocation to refer to something other than
  // the symbol itself, like a linker generated table. Since the address of
  // symbol is not relevant, we cannot replace the symbol with the
  // section and patch the difference in the addend.
  case MCSymbolRefExpr::VK_GOT:
  case MCSymbolRefExpr::VK_PLT:
  case MCSymbolRefExpr::VK_GOTPCREL:
  case MCSymbolRefExpr::VK_Mips_GOT:
  case MCSymbolRefExpr::VK_PPC_GOT_LO:
  case MCSymbolRefExpr::VK_PPC_GOT_HI:
  case MCSymbolRefExpr::VK_PPC_GOT_HA:
    return true;
  }

  // An undefined symbol is not in any section, so the relocation has to point
  // to the symbol itself.
  assert(Sym && "Expected a symbol");
  if (Sym->isUndefined())
    return true;

  unsigned Binding = Sym->getBinding();
  switch(Binding) {
  default:
    llvm_unreachable("Invalid Binding");
  case ELF::STB_LOCAL:
    break;
  case ELF::STB_WEAK:
    // If the symbol is weak, it might be overridden by a symbol in another
    // file. The relocation has to point to the symbol so that the linker
    // can update it.
    return true;
  case ELF::STB_GLOBAL:
    // Global ELF symbols can be preempted by the dynamic linker. The relocation
    // has to point to the symbol for a reason analogous to the STB_WEAK case.
    return true;
  }

  // If a relocation points to a mergeable section, we have to be careful.
  // If the offset is zero, a relocation with the section will encode the
  // same information. With a non-zero offset, the situation is different.
  // For example, a relocation can point 42 bytes past the end of a string.
  // If we change such a relocation to use the section, the linker would think
  // that it pointed to another string and subtracting 42 at runtime will
  // produce the wrong value.
  auto &Sec = cast<MCSectionELF>(Sym->getSection());
  unsigned Flags = Sec.getFlags();
  if (Flags & ELF::SHF_MERGE) {
    if (C != 0)
      return true;

    // It looks like gold has a bug (http://sourceware.org/PR16794) and can
    // only handle section relocations to mergeable sections if using RELA.
    if (!hasRelocationAddend())
      return true;
  }

  // Most TLS relocations use a got, so they need the symbol. Even those that
  // are just an offset (@tpoff), require a symbol in gold versions before
  // 5efeedf61e4fe720fd3e9a08e6c91c10abb66d42 (2014-09-26) which fixed
  // http://sourceware.org/PR16773.
  if (Flags & ELF::SHF_TLS)
    return true;

  // If the symbol is a thumb function the final relocation must set the lowest
  // bit. With a symbol that is done by just having the symbol have that bit
  // set, so we would lose the bit if we relocated with the section.
  // FIXME: We could use the section but add the bit to the relocation value.
  if (Asm.isThumbFunc(Sym))
    return true;

  if (TargetObjectWriter->needsRelocateWithSymbol(*Sym, Type))
    return true;
  return false;
}

// True if the assembler knows nothing about the final value of the symbol.
// This doesn't cover the comdat issues, since in those cases the assembler
// can at least know that all symbols in the section will move together.
static bool isWeak(const MCSymbolELF &Sym) {
  if (Sym.getType() == ELF::STT_GNU_IFUNC)
    return true;

  switch (Sym.getBinding()) {
  default:
    llvm_unreachable("Unknown binding");
  case ELF::STB_LOCAL:
    return false;
  case ELF::STB_GLOBAL:
    return false;
  case ELF::STB_WEAK:
  case ELF::STB_GNU_UNIQUE:
    return true;
  }
}

void ELFObjectWriter::recordRelocation(MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup, MCValue Target,
                                       bool &IsPCRel, uint64_t &FixedValue) {
  const MCSectionELF &FixupSection = cast<MCSectionELF>(*Fragment->getParent());
  uint64_t C = Target.getConstant();
  uint64_t FixupOffset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

  if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
    assert(RefB->getKind() == MCSymbolRefExpr::VK_None &&
           "Should not have constructed this");

    // Let A, B and C being the components of Target and R be the location of
    // the fixup. If the fixup is not pcrel, we want to compute (A - B + C).
    // If it is pcrel, we want to compute (A - B + C - R).

    // In general, ELF has no relocations for -B. It can only represent (A + C)
    // or (A + C - R). If B = R + K and the relocation is not pcrel, we can
    // replace B to implement it: (A - R - K + C)
    if (IsPCRel)
      Asm.getContext().reportFatalError(
          Fixup.getLoc(),
          "No relocation available to represent this relative expression");

    const auto &SymB = cast<MCSymbolELF>(RefB->getSymbol());

    if (SymB.isUndefined())
      Asm.getContext().reportFatalError(
          Fixup.getLoc(),
          Twine("symbol '") + SymB.getName() +
              "' can not be undefined in a subtraction expression");

    assert(!SymB.isAbsolute() && "Should have been folded");
    const MCSection &SecB = SymB.getSection();
    if (&SecB != &FixupSection)
      Asm.getContext().reportFatalError(
          Fixup.getLoc(), "Cannot represent a difference across sections");

    if (::isWeak(SymB))
      Asm.getContext().reportFatalError(
          Fixup.getLoc(), "Cannot represent a subtraction with a weak symbol");

    uint64_t SymBOffset = Layout.getSymbolOffset(SymB);
    uint64_t K = SymBOffset - FixupOffset;
    IsPCRel = true;
    C -= K;
  }

  // We either rejected the fixup or folded B into C at this point.
  const MCSymbolRefExpr *RefA = Target.getSymA();
  const auto *SymA = RefA ? cast<MCSymbolELF>(&RefA->getSymbol()) : nullptr;

  bool ViaWeakRef = false;
  if (SymA && SymA->isVariable()) {
    const MCExpr *Expr = SymA->getVariableValue();
    if (const auto *Inner = dyn_cast<MCSymbolRefExpr>(Expr)) {
      if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF) {
        SymA = cast<MCSymbolELF>(&Inner->getSymbol());
        ViaWeakRef = true;
      }
    }
  }

  unsigned Type = GetRelocType(Target, Fixup, IsPCRel);
  bool RelocateWithSymbol = shouldRelocateWithSymbol(Asm, RefA, SymA, C, Type);
  if (!RelocateWithSymbol && SymA && !SymA->isUndefined())
    C += Layout.getSymbolOffset(*SymA);

  uint64_t Addend = 0;
  if (hasRelocationAddend()) {
    Addend = C;
    C = 0;
  }

  FixedValue = C;

  if (!RelocateWithSymbol) {
    const MCSection *SecA =
        (SymA && !SymA->isUndefined()) ? &SymA->getSection() : nullptr;
    auto *ELFSec = cast_or_null<MCSectionELF>(SecA);
    const auto *SectionSymbol =
        ELFSec ? cast<MCSymbolELF>(ELFSec->getBeginSymbol()) : nullptr;
    if (SectionSymbol)
      SectionSymbol->setUsedInReloc();
    ELFRelocationEntry Rec(FixupOffset, SectionSymbol, Type, Addend);
    Relocations[&FixupSection].push_back(Rec);
    return;
  }

  if (SymA) {
    if (const MCSymbolELF *R = Renames.lookup(SymA))
      SymA = R;

    if (ViaWeakRef)
      SymA->setIsWeakrefUsedInReloc();
    else
      SymA->setUsedInReloc();
  }
  ELFRelocationEntry Rec(FixupOffset, SymA, Type, Addend);
  Relocations[&FixupSection].push_back(Rec);
  return;
}

bool ELFObjectWriter::isInSymtab(const MCAsmLayout &Layout,
                                 const MCSymbolELF &Symbol, bool Used,
                                 bool Renamed) {
  if (Symbol.isVariable()) {
    const MCExpr *Expr = Symbol.getVariableValue();
    if (const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(Expr)) {
      if (Ref->getKind() == MCSymbolRefExpr::VK_WEAKREF)
        return false;
    }
  }

  if (Used)
    return true;

  if (Renamed)
    return false;

  if (Symbol.isVariable() && Symbol.isUndefined()) {
    // FIXME: this is here just to diagnose the case of a var = commmon_sym.
    Layout.getBaseSymbol(Symbol);
    return false;
  }

  if (Symbol.isUndefined() && !Symbol.isBindingSet())
    return false;

  if (Symbol.isTemporary())
    return false;

  if (Symbol.getType() == ELF::STT_SECTION)
    return false;

  return true;
}

void ELFObjectWriter::computeSymbolTable(
    MCAssembler &Asm, const MCAsmLayout &Layout,
    const SectionIndexMapTy &SectionIndexMap, const RevGroupMapTy &RevGroupMap,
    SectionOffsetsTy &SectionOffsets) {
  MCContext &Ctx = Asm.getContext();
  SymbolTableWriter Writer(*this, is64Bit());

  // Symbol table
  unsigned EntrySize = is64Bit() ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;
  MCSectionELF *SymtabSection =
      Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0, EntrySize, "");
  SymtabSection->setAlignment(is64Bit() ? 8 : 4);
  SymbolTableIndex = addToSectionTable(SymtabSection);

  align(SymtabSection->getAlignment());
  uint64_t SecStart = OS.tell();

  // The first entry is the undefined symbol entry.
  Writer.writeSymbol(0, 0, 0, 0, 0, 0, false);

  std::vector<ELFSymbolData> LocalSymbolData;
  std::vector<ELFSymbolData> ExternalSymbolData;

  // Add the data for the symbols.
  bool HasLargeSectionIndex = false;
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &Symbol = cast<MCSymbolELF>(S);
    bool Used = Symbol.isUsedInReloc();
    bool WeakrefUsed = Symbol.isWeakrefUsedInReloc();
    bool isSignature = Symbol.isSignature();

    if (!isInSymtab(Layout, Symbol, Used || WeakrefUsed || isSignature,
                    Renames.count(&Symbol)))
      continue;

    if (Symbol.isTemporary() && Symbol.isUndefined())
      Ctx.reportFatalError(SMLoc(), "Undefined temporary");

    ELFSymbolData MSD;
    MSD.Symbol = cast<MCSymbolELF>(&Symbol);

    bool Local = Symbol.getBinding() == ELF::STB_LOCAL;
    assert(Local || !Symbol.isTemporary());

    if (Symbol.isAbsolute()) {
      MSD.SectionIndex = ELF::SHN_ABS;
    } else if (Symbol.isCommon()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_COMMON;
    } else if (Symbol.isUndefined()) {
      if (isSignature && !Used) {
        MSD.SectionIndex = RevGroupMap.lookup(&Symbol);
        if (MSD.SectionIndex >= ELF::SHN_LORESERVE)
          HasLargeSectionIndex = true;
      } else {
        MSD.SectionIndex = ELF::SHN_UNDEF;
      }
    } else {
      const MCSectionELF &Section =
          static_cast<const MCSectionELF &>(Symbol.getSection());
      MSD.SectionIndex = SectionIndexMap.lookup(&Section);
      assert(MSD.SectionIndex && "Invalid section index!");
      if (MSD.SectionIndex >= ELF::SHN_LORESERVE)
        HasLargeSectionIndex = true;
    }

    // The @@@ in symbol version is replaced with @ in undefined symbols and @@
    // in defined ones.
    //
    // FIXME: All name handling should be done before we get to the writer,
    // including dealing with GNU-style version suffixes.  Fixing this isn't
    // trivial.
    //
    // We thus have to be careful to not perform the symbol version replacement
    // blindly:
    //
    // The ELF format is used on Windows by the MCJIT engine.  Thus, on
    // Windows, the ELFObjectWriter can encounter symbols mangled using the MS
    // Visual Studio C++ name mangling scheme. Symbols mangled using the MSVC
    // C++ name mangling can legally have "@@@" as a sub-string. In that case,
    // the EFLObjectWriter should not interpret the "@@@" sub-string as
    // specifying GNU-style symbol versioning. The ELFObjectWriter therefore
    // checks for the MSVC C++ name mangling prefix which is either "?", "@?",
    // "__imp_?" or "__imp_@?".
    //
    // It would have been interesting to perform the MS mangling prefix check
    // only when the target triple is of the form *-pc-windows-elf. But, it
    // seems that this information is not easily accessible from the
    // ELFObjectWriter.
    StringRef Name = Symbol.getName();
    SmallString<32> Buf;
    if (!Name.startswith("?") && !Name.startswith("@?") &&
        !Name.startswith("__imp_?") && !Name.startswith("__imp_@?")) {
      // This symbol isn't following the MSVC C++ name mangling convention. We
      // can thus safely interpret the @@@ in symbol names as specifying symbol
      // versioning.
      size_t Pos = Name.find("@@@");
      if (Pos != StringRef::npos) {
        Buf += Name.substr(0, Pos);
        unsigned Skip = MSD.SectionIndex == ELF::SHN_UNDEF ? 2 : 1;
        Buf += Name.substr(Pos + Skip);
        Name = Buf;
      }
    }

    // Sections have their own string table
    if (Symbol.getType() != ELF::STT_SECTION)
      MSD.Name = StrTabBuilder.add(Name);

    if (Local)
      LocalSymbolData.push_back(MSD);
    else
      ExternalSymbolData.push_back(MSD);
  }

  // This holds the .symtab_shndx section index.
  unsigned SymtabShndxSectionIndex = 0;

  if (HasLargeSectionIndex) {
    MCSectionELF *SymtabShndxSection =
        Ctx.getELFSection(".symtab_shndxr", ELF::SHT_SYMTAB_SHNDX, 0, 4, "");
    SymtabShndxSectionIndex = addToSectionTable(SymtabShndxSection);
    SymtabShndxSection->setAlignment(4);
  }

  ArrayRef<std::string> FileNames = Asm.getFileNames();
  for (const std::string &Name : FileNames)
    StrTabBuilder.add(Name);

  StrTabBuilder.finalize(StringTableBuilder::ELF);

  for (const std::string &Name : FileNames)
    Writer.writeSymbol(StrTabBuilder.getOffset(Name),
                       ELF::STT_FILE | ELF::STB_LOCAL, 0, 0, ELF::STV_DEFAULT,
                       ELF::SHN_ABS, true);

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  unsigned Index = FileNames.size() + 1;

  for (ELFSymbolData &MSD : LocalSymbolData) {
    unsigned StringIndex = MSD.Symbol->getType() == ELF::STT_SECTION
                               ? 0
                               : StrTabBuilder.getOffset(MSD.Name);
    MSD.Symbol->setIndex(Index++);
    writeSymbol(Writer, StringIndex, MSD, Layout);
  }

  // Write the symbol table entries.
  LastLocalSymbolIndex = Index;

  for (ELFSymbolData &MSD : ExternalSymbolData) {
    unsigned StringIndex = StrTabBuilder.getOffset(MSD.Name);
    MSD.Symbol->setIndex(Index++);
    writeSymbol(Writer, StringIndex, MSD, Layout);
    assert(MSD.Symbol->getBinding() != ELF::STB_LOCAL);
  }

  uint64_t SecEnd = OS.tell();
  SectionOffsets[SymtabSection] = std::make_pair(SecStart, SecEnd);

  ArrayRef<uint32_t> ShndxIndexes = Writer.getShndxIndexes();
  if (ShndxIndexes.empty()) {
    assert(SymtabShndxSectionIndex == 0);
    return;
  }
  assert(SymtabShndxSectionIndex != 0);

  SecStart = OS.tell();
  const MCSectionELF *SymtabShndxSection =
      SectionTable[SymtabShndxSectionIndex - 1];
  for (uint32_t Index : ShndxIndexes)
    write(Index);
  SecEnd = OS.tell();
  SectionOffsets[SymtabShndxSection] = std::make_pair(SecStart, SecEnd);
}

MCSectionELF *
ELFObjectWriter::createRelocationSection(MCContext &Ctx,
                                         const MCSectionELF &Sec) {
  if (Relocations[&Sec].empty())
    return nullptr;

  const StringRef SectionName = Sec.getSectionName();
  std::string RelaSectionName = hasRelocationAddend() ? ".rela" : ".rel";
  RelaSectionName += SectionName;

  unsigned EntrySize;
  if (hasRelocationAddend())
    EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rela) : sizeof(ELF::Elf32_Rela);
  else
    EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rel) : sizeof(ELF::Elf32_Rel);

  unsigned Flags = 0;
  if (Sec.getFlags() & ELF::SHF_GROUP)
    Flags = ELF::SHF_GROUP;

  MCSectionELF *RelaSection = Ctx.createELFRelSection(
      RelaSectionName, hasRelocationAddend() ? ELF::SHT_RELA : ELF::SHT_REL,
      Flags, EntrySize, Sec.getGroup(), &Sec);
  RelaSection->setAlignment(is64Bit() ? 8 : 4);
  return RelaSection;
}

static SmallVector<char, 128>
getUncompressedData(const MCAsmLayout &Layout,
                    const MCSection::FragmentListType &Fragments) {
  SmallVector<char, 128> UncompressedData;
  for (const MCFragment &F : Fragments) {
    const SmallVectorImpl<char> *Contents;
    switch (F.getKind()) {
    case MCFragment::FT_Data:
      Contents = &cast<MCDataFragment>(F).getContents();
      break;
    case MCFragment::FT_Dwarf:
      Contents = &cast<MCDwarfLineAddrFragment>(F).getContents();
      break;
    case MCFragment::FT_DwarfFrame:
      Contents = &cast<MCDwarfCallFrameFragment>(F).getContents();
      break;
    default:
      llvm_unreachable(
          "Not expecting any other fragment types in a debug_* section");
    }
    UncompressedData.append(Contents->begin(), Contents->end());
  }
  return UncompressedData;
}

// Include the debug info compression header:
// "ZLIB" followed by 8 bytes representing the uncompressed size of the section,
// useful for consumers to preallocate a buffer to decompress into.
static bool
prependCompressionHeader(uint64_t Size,
                         SmallVectorImpl<char> &CompressedContents) {
  const StringRef Magic = "ZLIB";
  if (Size <= Magic.size() + sizeof(Size) + CompressedContents.size())
    return false;
  if (sys::IsLittleEndianHost)
    sys::swapByteOrder(Size);
  CompressedContents.insert(CompressedContents.begin(),
                            Magic.size() + sizeof(Size), 0);
  std::copy(Magic.begin(), Magic.end(), CompressedContents.begin());
  std::copy(reinterpret_cast<char *>(&Size),
            reinterpret_cast<char *>(&Size + 1),
            CompressedContents.begin() + Magic.size());
  return true;
}

void ELFObjectWriter::writeSectionData(const MCAssembler &Asm, MCSection &Sec,
                                       const MCAsmLayout &Layout) {
  MCSectionELF &Section = static_cast<MCSectionELF &>(Sec);
  StringRef SectionName = Section.getSectionName();

  // Compressing debug_frame requires handling alignment fragments which is
  // more work (possibly generalizing MCAssembler.cpp:writeFragment to allow
  // for writing to arbitrary buffers) for little benefit.
  if (!Asm.getContext().getAsmInfo()->compressDebugSections() ||
      !SectionName.startswith(".debug_") || SectionName == ".debug_frame") {
    Asm.writeSectionData(&Section, Layout);
    return;
  }

  // Gather the uncompressed data from all the fragments.
  const MCSection::FragmentListType &Fragments = Section.getFragmentList();
  SmallVector<char, 128> UncompressedData =
      getUncompressedData(Layout, Fragments);

  SmallVector<char, 128> CompressedContents;
  zlib::Status Success = zlib::compress(
      StringRef(UncompressedData.data(), UncompressedData.size()),
      CompressedContents);
  if (Success != zlib::StatusOK) {
    Asm.writeSectionData(&Section, Layout);
    return;
  }

  if (!prependCompressionHeader(UncompressedData.size(), CompressedContents)) {
    Asm.writeSectionData(&Section, Layout);
    return;
  }
  Asm.getContext().renameELFSection(&Section,
                                    (".z" + SectionName.drop_front(1)).str());
  OS << CompressedContents;
}

void ELFObjectWriter::WriteSecHdrEntry(uint32_t Name, uint32_t Type,
                                       uint64_t Flags, uint64_t Address,
                                       uint64_t Offset, uint64_t Size,
                                       uint32_t Link, uint32_t Info,
                                       uint64_t Alignment,
                                       uint64_t EntrySize) {
  write32(Name);        // sh_name: index into string table
  write32(Type);        // sh_type
  WriteWord(Flags);     // sh_flags
  WriteWord(Address);   // sh_addr
  WriteWord(Offset);    // sh_offset
  WriteWord(Size);      // sh_size
  write32(Link);        // sh_link
  write32(Info);        // sh_info
  WriteWord(Alignment); // sh_addralign
  WriteWord(EntrySize); // sh_entsize
}

void ELFObjectWriter::writeRelocations(const MCAssembler &Asm,
                                       const MCSectionELF &Sec) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[&Sec];

  // Sort the relocation entries. Most targets just sort by Offset, but some
  // (e.g., MIPS) have additional constraints.
  TargetObjectWriter->sortRelocs(Asm, Relocs);

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    const ELFRelocationEntry &Entry = Relocs[e - i - 1];
    unsigned Index = Entry.Symbol ? Entry.Symbol->getIndex() : 0;

    if (is64Bit()) {
      write(Entry.Offset);
      if (TargetObjectWriter->isN64()) {
        write(uint32_t(Index));

        write(TargetObjectWriter->getRSsym(Entry.Type));
        write(TargetObjectWriter->getRType3(Entry.Type));
        write(TargetObjectWriter->getRType2(Entry.Type));
        write(TargetObjectWriter->getRType(Entry.Type));
      } else {
        struct ELF::Elf64_Rela ERE64;
        ERE64.setSymbolAndType(Index, Entry.Type);
        write(ERE64.r_info);
      }
      if (hasRelocationAddend())
        write(Entry.Addend);
    } else {
      write(uint32_t(Entry.Offset));

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(Index, Entry.Type);
      write(ERE32.r_info);

      if (hasRelocationAddend())
        write(uint32_t(Entry.Addend));
    }
  }
}

const MCSectionELF *ELFObjectWriter::createStringTable(MCContext &Ctx) {
  const MCSectionELF *StrtabSection = SectionTable[StringTableIndex - 1];
  OS << StrTabBuilder.data();
  return StrtabSection;
}

void ELFObjectWriter::writeSection(const SectionIndexMapTy &SectionIndexMap,
                                   uint32_t GroupSymbolIndex, uint64_t Offset,
                                   uint64_t Size, const MCSectionELF &Section) {
  uint64_t sh_link = 0;
  uint64_t sh_info = 0;

  switch(Section.getType()) {
  default:
    // Nothing to do.
    break;

  case ELF::SHT_DYNAMIC:
    llvm_unreachable("SHT_DYNAMIC in a relocatable object");

  case ELF::SHT_REL:
  case ELF::SHT_RELA: {
    sh_link = SymbolTableIndex;
    assert(sh_link && ".symtab not found");
    const MCSectionELF *InfoSection = Section.getAssociatedSection();
    sh_info = SectionIndexMap.lookup(InfoSection);
    break;
  }

  case ELF::SHT_SYMTAB:
  case ELF::SHT_DYNSYM:
    sh_link = StringTableIndex;
    sh_info = LastLocalSymbolIndex;
    break;

  case ELF::SHT_SYMTAB_SHNDX:
    sh_link = SymbolTableIndex;
    break;

  case ELF::SHT_GROUP:
    sh_link = SymbolTableIndex;
    sh_info = GroupSymbolIndex;
    break;
  }

  if (TargetObjectWriter->getEMachine() == ELF::EM_ARM &&
      Section.getType() == ELF::SHT_ARM_EXIDX)
    sh_link = SectionIndexMap.lookup(Section.getAssociatedSection());

  WriteSecHdrEntry(StrTabBuilder.getOffset(Section.getSectionName()),
                   Section.getType(), Section.getFlags(), 0, Offset, Size,
                   sh_link, sh_info, Section.getAlignment(),
                   Section.getEntrySize());
}

void ELFObjectWriter::writeSectionHeader(
    const MCAsmLayout &Layout, const SectionIndexMapTy &SectionIndexMap,
    const SectionOffsetsTy &SectionOffsets) {
  const unsigned NumSections = SectionTable.size();

  // Null section first.
  uint64_t FirstSectionSize =
      (NumSections + 1) >= ELF::SHN_LORESERVE ? NumSections + 1 : 0;
  WriteSecHdrEntry(0, 0, 0, 0, 0, FirstSectionSize, 0, 0, 0, 0);

  for (const MCSectionELF *Section : SectionTable) {
    uint32_t GroupSymbolIndex;
    unsigned Type = Section->getType();
    if (Type != ELF::SHT_GROUP)
      GroupSymbolIndex = 0;
    else
      GroupSymbolIndex = Section->getGroup()->getIndex();

    const std::pair<uint64_t, uint64_t> &Offsets =
        SectionOffsets.find(Section)->second;
    uint64_t Size;
    if (Type == ELF::SHT_NOBITS)
      Size = Layout.getSectionAddressSize(Section);
    else
      Size = Offsets.second - Offsets.first;

    writeSection(SectionIndexMap, GroupSymbolIndex, Offsets.first, Size,
                 *Section);
  }
}

void ELFObjectWriter::writeObject(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  MCContext &Ctx = Asm.getContext();
  MCSectionELF *StrtabSection =
      Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0);
  StringTableIndex = addToSectionTable(StrtabSection);

  RevGroupMapTy RevGroupMap;
  SectionIndexMapTy SectionIndexMap;

  std::map<const MCSymbol *, std::vector<const MCSectionELF *>> GroupMembers;

  // Write out the ELF header ...
  writeHeader(Asm);

  // ... then the sections ...
  SectionOffsetsTy SectionOffsets;
  std::vector<MCSectionELF *> Groups;
  std::vector<MCSectionELF *> Relocations;
  for (MCSection &Sec : Asm) {
    MCSectionELF &Section = static_cast<MCSectionELF &>(Sec);

    align(Section.getAlignment());

    // Remember the offset into the file for this section.
    uint64_t SecStart = OS.tell();

    const MCSymbolELF *SignatureSymbol = Section.getGroup();
    writeSectionData(Asm, Section, Layout);

    uint64_t SecEnd = OS.tell();
    SectionOffsets[&Section] = std::make_pair(SecStart, SecEnd);

    MCSectionELF *RelSection = createRelocationSection(Ctx, Section);

    if (SignatureSymbol) {
      Asm.registerSymbol(*SignatureSymbol);
      unsigned &GroupIdx = RevGroupMap[SignatureSymbol];
      if (!GroupIdx) {
        MCSectionELF *Group = Ctx.createELFGroupSection(SignatureSymbol);
        GroupIdx = addToSectionTable(Group);
        Group->setAlignment(4);
        Groups.push_back(Group);
      }
      std::vector<const MCSectionELF *> &Members =
          GroupMembers[SignatureSymbol];
      Members.push_back(&Section);
      if (RelSection)
        Members.push_back(RelSection);
    }

    SectionIndexMap[&Section] = addToSectionTable(&Section);
    if (RelSection) {
      SectionIndexMap[RelSection] = addToSectionTable(RelSection);
      Relocations.push_back(RelSection);
    }
  }

  for (MCSectionELF *Group : Groups) {
    align(Group->getAlignment());

    // Remember the offset into the file for this section.
    uint64_t SecStart = OS.tell();

    const MCSymbol *SignatureSymbol = Group->getGroup();
    assert(SignatureSymbol);
    write(uint32_t(ELF::GRP_COMDAT));
    for (const MCSectionELF *Member : GroupMembers[SignatureSymbol]) {
      uint32_t SecIndex = SectionIndexMap.lookup(Member);
      write(SecIndex);
    }

    uint64_t SecEnd = OS.tell();
    SectionOffsets[Group] = std::make_pair(SecStart, SecEnd);
  }

  // Compute symbol table information.
  computeSymbolTable(Asm, Layout, SectionIndexMap, RevGroupMap, SectionOffsets);

  for (MCSectionELF *RelSection : Relocations) {
    align(RelSection->getAlignment());

    // Remember the offset into the file for this section.
    uint64_t SecStart = OS.tell();

    writeRelocations(Asm, *RelSection->getAssociatedSection());

    uint64_t SecEnd = OS.tell();
    SectionOffsets[RelSection] = std::make_pair(SecStart, SecEnd);
  }

  {
    uint64_t SecStart = OS.tell();
    const MCSectionELF *Sec = createStringTable(Ctx);
    uint64_t SecEnd = OS.tell();
    SectionOffsets[Sec] = std::make_pair(SecStart, SecEnd);
  }

  uint64_t NaturalAlignment = is64Bit() ? 8 : 4;
  align(NaturalAlignment);

  const unsigned SectionHeaderOffset = OS.tell();

  // ... then the section header table ...
  writeSectionHeader(Layout, SectionIndexMap, SectionOffsets);

  uint16_t NumSections = (SectionTable.size() + 1 >= ELF::SHN_LORESERVE)
                             ? (uint16_t)ELF::SHN_UNDEF
                             : SectionTable.size() + 1;
  if (sys::IsLittleEndianHost != IsLittleEndian)
    sys::swapByteOrder(NumSections);
  unsigned NumSectionsOffset;

  if (is64Bit()) {
    uint64_t Val = SectionHeaderOffset;
    if (sys::IsLittleEndianHost != IsLittleEndian)
      sys::swapByteOrder(Val);
    OS.pwrite(reinterpret_cast<char *>(&Val), sizeof(Val),
              offsetof(ELF::Elf64_Ehdr, e_shoff));
    NumSectionsOffset = offsetof(ELF::Elf64_Ehdr, e_shnum);
  } else {
    uint32_t Val = SectionHeaderOffset;
    if (sys::IsLittleEndianHost != IsLittleEndian)
      sys::swapByteOrder(Val);
    OS.pwrite(reinterpret_cast<char *>(&Val), sizeof(Val),
              offsetof(ELF::Elf32_Ehdr, e_shoff));
    NumSectionsOffset = offsetof(ELF::Elf32_Ehdr, e_shnum);
  }
  OS.pwrite(reinterpret_cast<char *>(&NumSections), sizeof(NumSections),
            NumSectionsOffset);
}

bool ELFObjectWriter::isSymbolRefDifferenceFullyResolvedImpl(
    const MCAssembler &Asm, const MCSymbol &SA, const MCFragment &FB,
    bool InSet, bool IsPCRel) const {
  const auto &SymA = cast<MCSymbolELF>(SA);
  if (IsPCRel) {
    assert(!InSet);
    if (::isWeak(SymA))
      return false;
  }
  return MCObjectWriter::isSymbolRefDifferenceFullyResolvedImpl(Asm, SymA, FB,
                                                                InSet, IsPCRel);
}

bool ELFObjectWriter::isWeak(const MCSymbol &S) const {
  const auto &Sym = cast<MCSymbolELF>(S);
  if (::isWeak(Sym))
    return true;

  // It is invalid to replace a reference to a global in a comdat
  // with a reference to a local since out of comdat references
  // to a local are forbidden.
  // We could try to return false for more cases, like the reference
  // being in the same comdat or Sym being an alias to another global,
  // but it is not clear if it is worth the effort.
  if (Sym.getBinding() != ELF::STB_GLOBAL)
    return false;

  if (!Sym.isInSection())
    return false;

  const auto &Sec = cast<MCSectionELF>(Sym.getSection());
  return Sec.getGroup();
}

MCObjectWriter *llvm::createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                            raw_pwrite_stream &OS,
                                            bool IsLittleEndian) {
  return new ELFObjectWriter(MOTW, OS, IsLittleEndian);
}
