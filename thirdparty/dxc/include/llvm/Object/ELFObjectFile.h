//===- ELFObjectFile.h - ELF object file implementation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFObjectFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELFOBJECTFILE_H
#define LLVM_OBJECT_ELFOBJECTFILE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>

namespace llvm {
namespace object {

class elf_symbol_iterator;
class ELFSymbolRef;
class ELFRelocationRef;

class ELFObjectFileBase : public ObjectFile {
  friend class ELFSymbolRef;
  friend class ELFSectionRef;
  friend class ELFRelocationRef;

protected:
  ELFObjectFileBase(unsigned int Type, MemoryBufferRef Source);

  virtual uint64_t getSymbolSize(DataRefImpl Symb) const = 0;
  virtual uint8_t getSymbolOther(DataRefImpl Symb) const = 0;
  virtual uint8_t getSymbolELFType(DataRefImpl Symb) const = 0;

  virtual uint32_t getSectionType(DataRefImpl Sec) const = 0;
  virtual uint64_t getSectionFlags(DataRefImpl Sec) const = 0;

  virtual ErrorOr<int64_t> getRelocationAddend(DataRefImpl Rel) const = 0;
public:

  typedef iterator_range<elf_symbol_iterator> elf_symbol_iterator_range;
  virtual elf_symbol_iterator_range getDynamicSymbolIterators() const = 0;

  elf_symbol_iterator_range symbols() const;

  static inline bool classof(const Binary *v) { return v->isELF(); }
};

class ELFSectionRef : public SectionRef {
public:
  ELFSectionRef(const SectionRef &B) : SectionRef(B) {
    assert(isa<ELFObjectFileBase>(SectionRef::getObject()));
  }

  const ELFObjectFileBase *getObject() const {
    return cast<ELFObjectFileBase>(SectionRef::getObject());
  }

  uint32_t getType() const {
    return getObject()->getSectionType(getRawDataRefImpl());
  }

  uint64_t getFlags() const {
    return getObject()->getSectionFlags(getRawDataRefImpl());
  }
};

class elf_section_iterator : public section_iterator {
public:
  elf_section_iterator(const section_iterator &B) : section_iterator(B) {
    assert(isa<ELFObjectFileBase>(B->getObject()));
  }

  const ELFSectionRef *operator->() const {
    return static_cast<const ELFSectionRef *>(section_iterator::operator->());
  }

  const ELFSectionRef &operator*() const {
    return static_cast<const ELFSectionRef &>(section_iterator::operator*());
  }
};

class ELFSymbolRef : public SymbolRef {
public:
  ELFSymbolRef(const SymbolRef &B) : SymbolRef(B) {
    assert(isa<ELFObjectFileBase>(SymbolRef::getObject()));
  }

  const ELFObjectFileBase *getObject() const {
    return cast<ELFObjectFileBase>(BasicSymbolRef::getObject());
  }

  uint64_t getSize() const {
    return getObject()->getSymbolSize(getRawDataRefImpl());
  }

  uint8_t getOther() const {
    return getObject()->getSymbolOther(getRawDataRefImpl());
  }

  uint8_t getELFType() const {
    return getObject()->getSymbolELFType(getRawDataRefImpl());
  }
};

class elf_symbol_iterator : public symbol_iterator {
public:
  elf_symbol_iterator(const basic_symbol_iterator &B)
      : symbol_iterator(SymbolRef(B->getRawDataRefImpl(),
                                  cast<ELFObjectFileBase>(B->getObject()))) {}

  const ELFSymbolRef *operator->() const {
    return static_cast<const ELFSymbolRef *>(symbol_iterator::operator->());
  }

  const ELFSymbolRef &operator*() const {
    return static_cast<const ELFSymbolRef &>(symbol_iterator::operator*());
  }
};

class ELFRelocationRef : public RelocationRef {
public:
  ELFRelocationRef(const RelocationRef &B) : RelocationRef(B) {
    assert(isa<ELFObjectFileBase>(RelocationRef::getObject()));
  }

  const ELFObjectFileBase *getObject() const {
    return cast<ELFObjectFileBase>(RelocationRef::getObject());
  }

  ErrorOr<int64_t> getAddend() const {
    return getObject()->getRelocationAddend(getRawDataRefImpl());
  }
};

class elf_relocation_iterator : public relocation_iterator {
public:
  elf_relocation_iterator(const relocation_iterator &B)
      : relocation_iterator(RelocationRef(
            B->getRawDataRefImpl(), cast<ELFObjectFileBase>(B->getObject()))) {}

  const ELFRelocationRef *operator->() const {
    return static_cast<const ELFRelocationRef *>(
        relocation_iterator::operator->());
  }

  const ELFRelocationRef &operator*() const {
    return static_cast<const ELFRelocationRef &>(
        relocation_iterator::operator*());
  }
};

inline ELFObjectFileBase::elf_symbol_iterator_range
ELFObjectFileBase::symbols() const {
  return elf_symbol_iterator_range(symbol_begin(), symbol_end());
}

template <class ELFT> class ELFObjectFile : public ELFObjectFileBase {
  uint64_t getSymbolSize(DataRefImpl Sym) const override;

public:
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)

  typedef typename ELFFile<ELFT>::uintX_t uintX_t;

  typedef typename ELFFile<ELFT>::Elf_Sym Elf_Sym;
  typedef typename ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  typedef typename ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename ELFFile<ELFT>::Elf_Rela Elf_Rela;
  typedef typename ELFFile<ELFT>::Elf_Dyn Elf_Dyn;

  typedef typename ELFFile<ELFT>::Elf_Dyn_Iter Elf_Dyn_Iter;

protected:
  ELFFile<ELFT> EF;

  void moveSymbolNext(DataRefImpl &Symb) const override;
  ErrorOr<StringRef> getSymbolName(DataRefImpl Symb) const override;
  ErrorOr<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint32_t getSymbolAlignment(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  uint32_t getSymbolFlags(DataRefImpl Symb) const override;
  uint8_t getSymbolOther(DataRefImpl Symb) const override;
  uint8_t getSymbolELFType(DataRefImpl Symb) const override;
  SymbolRef::Type getSymbolType(DataRefImpl Symb) const override;
  section_iterator getSymbolSection(const Elf_Sym *Symb) const;
  std::error_code getSymbolSection(DataRefImpl Symb,
                                   section_iterator &Res) const override;

  void moveSectionNext(DataRefImpl &Sec) const override;
  std::error_code getSectionName(DataRefImpl Sec,
                                 StringRef &Res) const override;
  uint64_t getSectionAddress(DataRefImpl Sec) const override;
  uint64_t getSectionSize(DataRefImpl Sec) const override;
  std::error_code getSectionContents(DataRefImpl Sec,
                                     StringRef &Res) const override;
  uint64_t getSectionAlignment(DataRefImpl Sec) const override;
  bool isSectionText(DataRefImpl Sec) const override;
  bool isSectionData(DataRefImpl Sec) const override;
  bool isSectionBSS(DataRefImpl Sec) const override;
  bool isSectionVirtual(DataRefImpl Sec) const override;
  relocation_iterator section_rel_begin(DataRefImpl Sec) const override;
  relocation_iterator section_rel_end(DataRefImpl Sec) const override;
  section_iterator getRelocatedSection(DataRefImpl Sec) const override;

  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;

  uint32_t getSectionType(DataRefImpl Sec) const override;
  uint64_t getSectionFlags(DataRefImpl Sec) const override;
  StringRef getRelocationTypeName(uint32_t Type) const;

  /// \brief Get the relocation section that contains \a Rel.
  const Elf_Shdr *getRelSection(DataRefImpl Rel) const {
    return *EF.getSection(Rel.d.a);
  }

  const Elf_Sym *toELFSymIter(DataRefImpl Sym) const {
    return EF.template getEntry<Elf_Sym>(Sym.d.a, Sym.d.b);
  }

  DataRefImpl toDRI(const Elf_Shdr *SymTable, unsigned SymbolNum) const {
    DataRefImpl DRI;
    if (!SymTable) {
      DRI.d.a = 0;
      DRI.d.b = 0;
      return DRI;
    }
    assert(SymTable->sh_type == ELF::SHT_SYMTAB ||
           SymTable->sh_type == ELF::SHT_DYNSYM);

    uintptr_t SHT = reinterpret_cast<uintptr_t>(EF.section_begin());
    unsigned SymTableIndex =
        (reinterpret_cast<uintptr_t>(SymTable) - SHT) / sizeof(Elf_Shdr);

    DRI.d.a = SymTableIndex;
    DRI.d.b = SymbolNum;
    return DRI;
  }

  const Elf_Shdr *toELFShdrIter(DataRefImpl Sec) const {
    return reinterpret_cast<const Elf_Shdr *>(Sec.p);
  }

  DataRefImpl toDRI(const Elf_Shdr *Sec) const {
    DataRefImpl DRI;
    DRI.p = reinterpret_cast<uintptr_t>(Sec);
    return DRI;
  }

  DataRefImpl toDRI(Elf_Dyn_Iter Dyn) const {
    DataRefImpl DRI;
    DRI.p = reinterpret_cast<uintptr_t>(Dyn.get());
    return DRI;
  }

  bool isExportedToOtherDSO(const Elf_Sym *ESym) const {
    unsigned char Binding = ESym->getBinding();
    unsigned char Visibility = ESym->getVisibility();

    // A symbol is exported if its binding is either GLOBAL or WEAK, and its
    // visibility is either DEFAULT or PROTECTED. All other symbols are not
    // exported.
    if ((Binding == ELF::STB_GLOBAL || Binding == ELF::STB_WEAK) &&
        (Visibility == ELF::STV_DEFAULT || Visibility == ELF::STV_PROTECTED))
      return true;

    return false;
  }

  // This flag is used for classof, to distinguish ELFObjectFile from
  // its subclass. If more subclasses will be created, this flag will
  // have to become an enum.
  bool isDyldELFObject;

public:
  ELFObjectFile(MemoryBufferRef Object, std::error_code &EC);

  const Elf_Rel *getRel(DataRefImpl Rel) const;
  const Elf_Rela *getRela(DataRefImpl Rela) const;

  const Elf_Sym *getSymbol(DataRefImpl Symb) const;

  basic_symbol_iterator symbol_begin_impl() const override;
  basic_symbol_iterator symbol_end_impl() const override;

  elf_symbol_iterator dynamic_symbol_begin() const;
  elf_symbol_iterator dynamic_symbol_end() const;

  section_iterator section_begin() const override;
  section_iterator section_end() const override;

  ErrorOr<int64_t> getRelocationAddend(DataRefImpl Rel) const override;

  uint8_t getBytesInAddress() const override;
  StringRef getFileFormatName() const override;
  unsigned getArch() const override;
  StringRef getLoadName() const;

  std::error_code getPlatformFlags(unsigned &Result) const override {
    Result = EF.getHeader()->e_flags;
    return std::error_code();
  }

  const ELFFile<ELFT> *getELFFile() const { return &EF; }

  bool isDyldType() const { return isDyldELFObject; }
  static inline bool classof(const Binary *v) {
    return v->getType() == getELFType(ELFT::TargetEndianness == support::little,
                                      ELFT::Is64Bits);
  }

  elf_symbol_iterator_range getDynamicSymbolIterators() const override;

  bool isRelocatableObject() const override;
};

typedef ELFObjectFile<ELFType<support::little, false>> ELF32LEObjectFile;
typedef ELFObjectFile<ELFType<support::little, true>> ELF64LEObjectFile;
typedef ELFObjectFile<ELFType<support::big, false>> ELF32BEObjectFile;
typedef ELFObjectFile<ELFType<support::big, true>> ELF64BEObjectFile;

template <class ELFT>
void ELFObjectFile<ELFT>::moveSymbolNext(DataRefImpl &Sym) const {
  ++Sym.d.b;
}

template <class ELFT>
ErrorOr<StringRef> ELFObjectFile<ELFT>::getSymbolName(DataRefImpl Sym) const {
  const Elf_Sym *ESym = toELFSymIter(Sym);
  const Elf_Shdr *SymTableSec = *EF.getSection(Sym.d.a);
  const Elf_Shdr *StringTableSec = *EF.getSection(SymTableSec->sh_link);
  StringRef SymTable = *EF.getStringTable(StringTableSec);
  return ESym->getName(SymTable);
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getSectionFlags(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_flags;
}

template <class ELFT>
uint32_t ELFObjectFile<ELFT>::getSectionType(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_type;
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getSymbolValueImpl(DataRefImpl Symb) const {
  const Elf_Sym *ESym = getSymbol(Symb);
  uint64_t Ret = ESym->st_value;
  if (ESym->st_shndx == ELF::SHN_ABS)
    return Ret;

  const Elf_Ehdr *Header = EF.getHeader();
  // Clear the ARM/Thumb or microMIPS indicator flag.
  if ((Header->e_machine == ELF::EM_ARM || Header->e_machine == ELF::EM_MIPS) &&
      ESym->getType() == ELF::STT_FUNC)
    Ret &= ~1;

  return Ret;
}

template <class ELFT>
ErrorOr<uint64_t>
ELFObjectFile<ELFT>::getSymbolAddress(DataRefImpl Symb) const {
  uint64_t Result = getSymbolValue(Symb);
  const Elf_Sym *ESym = getSymbol(Symb);
  switch (ESym->st_shndx) {
  case ELF::SHN_COMMON:
  case ELF::SHN_UNDEF:
  case ELF::SHN_ABS:
    return Result;
  }

  const Elf_Ehdr *Header = EF.getHeader();

  if (Header->e_type == ELF::ET_REL) {
    ErrorOr<const Elf_Shdr *> SectionOrErr = EF.getSection(ESym);
    if (std::error_code EC = SectionOrErr.getError())
      return EC;
    const Elf_Shdr *Section = *SectionOrErr;
    if (Section)
      Result += Section->sh_addr;
  }

  return Result;
}

template <class ELFT>
uint32_t ELFObjectFile<ELFT>::getSymbolAlignment(DataRefImpl Symb) const {
  const Elf_Sym *Sym = toELFSymIter(Symb);
  if (Sym->st_shndx == ELF::SHN_COMMON)
    return Sym->st_value;
  return 0;
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getSymbolSize(DataRefImpl Sym) const {
  return toELFSymIter(Sym)->st_size;
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  return toELFSymIter(Symb)->st_size;
}

template <class ELFT>
uint8_t ELFObjectFile<ELFT>::getSymbolOther(DataRefImpl Symb) const {
  return toELFSymIter(Symb)->st_other;
}

template <class ELFT>
uint8_t ELFObjectFile<ELFT>::getSymbolELFType(DataRefImpl Symb) const {
  return toELFSymIter(Symb)->getType();
}

template <class ELFT>
SymbolRef::Type ELFObjectFile<ELFT>::getSymbolType(DataRefImpl Symb) const {
  const Elf_Sym *ESym = getSymbol(Symb);

  switch (ESym->getType()) {
  case ELF::STT_NOTYPE:
    return SymbolRef::ST_Unknown;
  case ELF::STT_SECTION:
    return SymbolRef::ST_Debug;
  case ELF::STT_FILE:
    return SymbolRef::ST_File;
  case ELF::STT_FUNC:
    return SymbolRef::ST_Function;
  case ELF::STT_OBJECT:
  case ELF::STT_COMMON:
  case ELF::STT_TLS:
    return SymbolRef::ST_Data;
  default:
    return SymbolRef::ST_Other;
  }
}

template <class ELFT>
uint32_t ELFObjectFile<ELFT>::getSymbolFlags(DataRefImpl Sym) const {
  const Elf_Sym *ESym = toELFSymIter(Sym);

  uint32_t Result = SymbolRef::SF_None;

  if (ESym->getBinding() != ELF::STB_LOCAL)
    Result |= SymbolRef::SF_Global;

  if (ESym->getBinding() == ELF::STB_WEAK)
    Result |= SymbolRef::SF_Weak;

  if (ESym->st_shndx == ELF::SHN_ABS)
    Result |= SymbolRef::SF_Absolute;

  if (ESym->getType() == ELF::STT_FILE || ESym->getType() == ELF::STT_SECTION ||
      ESym == EF.symbol_begin() || ESym == EF.dynamic_symbol_begin())
    Result |= SymbolRef::SF_FormatSpecific;

  if (EF.getHeader()->e_machine == ELF::EM_ARM) {
    if (ErrorOr<StringRef> NameOrErr = getSymbolName(Sym)) {
      StringRef Name = *NameOrErr;
      if (Name.startswith("$d") || Name.startswith("$t") ||
          Name.startswith("$a"))
        Result |= SymbolRef::SF_FormatSpecific;
    }
  }

  if (ESym->st_shndx == ELF::SHN_UNDEF)
    Result |= SymbolRef::SF_Undefined;

  if (ESym->getType() == ELF::STT_COMMON || ESym->st_shndx == ELF::SHN_COMMON)
    Result |= SymbolRef::SF_Common;

  if (isExportedToOtherDSO(ESym))
    Result |= SymbolRef::SF_Exported;

  if (ESym->getVisibility() == ELF::STV_HIDDEN)
    Result |= SymbolRef::SF_Hidden;

  return Result;
}

template <class ELFT>
section_iterator
ELFObjectFile<ELFT>::getSymbolSection(const Elf_Sym *ESym) const {
  ErrorOr<const Elf_Shdr *> ESecOrErr = EF.getSection(ESym);
  if (std::error_code EC = ESecOrErr.getError())
    report_fatal_error(EC.message());

  const Elf_Shdr *ESec = *ESecOrErr;
  if (!ESec)
    return section_end();

  DataRefImpl Sec;
  Sec.p = reinterpret_cast<intptr_t>(ESec);
  return section_iterator(SectionRef(Sec, this));
}

template <class ELFT>
std::error_code
ELFObjectFile<ELFT>::getSymbolSection(DataRefImpl Symb,
                                      section_iterator &Res) const {
  Res = getSymbolSection(getSymbol(Symb));
  return std::error_code();
}

template <class ELFT>
void ELFObjectFile<ELFT>::moveSectionNext(DataRefImpl &Sec) const {
  const Elf_Shdr *ESec = toELFShdrIter(Sec);
  Sec = toDRI(++ESec);
}

template <class ELFT>
std::error_code ELFObjectFile<ELFT>::getSectionName(DataRefImpl Sec,
                                                    StringRef &Result) const {
  ErrorOr<StringRef> Name = EF.getSectionName(&*toELFShdrIter(Sec));
  if (!Name)
    return Name.getError();
  Result = *Name;
  return std::error_code();
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getSectionAddress(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_addr;
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getSectionSize(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_size;
}

template <class ELFT>
std::error_code
ELFObjectFile<ELFT>::getSectionContents(DataRefImpl Sec,
                                        StringRef &Result) const {
  const Elf_Shdr *EShdr = toELFShdrIter(Sec);
  Result = StringRef((const char *)base() + EShdr->sh_offset, EShdr->sh_size);
  return std::error_code();
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getSectionAlignment(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_addralign;
}

template <class ELFT>
bool ELFObjectFile<ELFT>::isSectionText(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_flags & ELF::SHF_EXECINSTR;
}

template <class ELFT>
bool ELFObjectFile<ELFT>::isSectionData(DataRefImpl Sec) const {
  const Elf_Shdr *EShdr = toELFShdrIter(Sec);
  return EShdr->sh_flags & (ELF::SHF_ALLOC | ELF::SHF_WRITE) &&
         EShdr->sh_type == ELF::SHT_PROGBITS;
}

template <class ELFT>
bool ELFObjectFile<ELFT>::isSectionBSS(DataRefImpl Sec) const {
  const Elf_Shdr *EShdr = toELFShdrIter(Sec);
  return EShdr->sh_flags & (ELF::SHF_ALLOC | ELF::SHF_WRITE) &&
         EShdr->sh_type == ELF::SHT_NOBITS;
}

template <class ELFT>
bool ELFObjectFile<ELFT>::isSectionVirtual(DataRefImpl Sec) const {
  return toELFShdrIter(Sec)->sh_type == ELF::SHT_NOBITS;
}

template <class ELFT>
relocation_iterator
ELFObjectFile<ELFT>::section_rel_begin(DataRefImpl Sec) const {
  DataRefImpl RelData;
  uintptr_t SHT = reinterpret_cast<uintptr_t>(EF.section_begin());
  RelData.d.a = (Sec.p - SHT) / EF.getHeader()->e_shentsize;
  RelData.d.b = 0;

  const Elf_Shdr *S = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  if (S->sh_type != ELF::SHT_RELA && S->sh_type != ELF::SHT_REL)
    return relocation_iterator(RelocationRef(RelData, this));

  const Elf_Shdr *RelSec = getRelSection(RelData);
  ErrorOr<const Elf_Shdr *> SymSecOrErr = EF.getSection(RelSec->sh_link);
  if (std::error_code EC = SymSecOrErr.getError())
    report_fatal_error(EC.message());
  const Elf_Shdr *SymSec = *SymSecOrErr;
  uint32_t SymSecType = SymSec->sh_type;
  if (SymSecType != ELF::SHT_SYMTAB && SymSecType != ELF::SHT_DYNSYM)
    report_fatal_error("Invalid symbol table section type!");
  if (SymSecType == ELF::SHT_DYNSYM)
    RelData.d.b = 1;

  return relocation_iterator(RelocationRef(RelData, this));
}

template <class ELFT>
relocation_iterator
ELFObjectFile<ELFT>::section_rel_end(DataRefImpl Sec) const {
  const Elf_Shdr *S = reinterpret_cast<const Elf_Shdr *>(Sec.p);
  relocation_iterator Begin = section_rel_begin(Sec);
  if (S->sh_type != ELF::SHT_RELA && S->sh_type != ELF::SHT_REL)
    return Begin;
  DataRefImpl RelData = Begin->getRawDataRefImpl();
  RelData.d.b += (S->sh_size / S->sh_entsize) << 1;
  return relocation_iterator(RelocationRef(RelData, this));
}

template <class ELFT>
section_iterator
ELFObjectFile<ELFT>::getRelocatedSection(DataRefImpl Sec) const {
  if (EF.getHeader()->e_type != ELF::ET_REL)
    return section_end();

  const Elf_Shdr *EShdr = toELFShdrIter(Sec);
  uintX_t Type = EShdr->sh_type;
  if (Type != ELF::SHT_REL && Type != ELF::SHT_RELA)
    return section_end();

  ErrorOr<const Elf_Shdr *> R = EF.getSection(EShdr->sh_info);
  if (std::error_code EC = R.getError())
    report_fatal_error(EC.message());
  return section_iterator(SectionRef(toDRI(*R), this));
}

// Relocations
template <class ELFT>
void ELFObjectFile<ELFT>::moveRelocationNext(DataRefImpl &Rel) const {
  Rel.d.b += 2;
}

template <class ELFT>
symbol_iterator
ELFObjectFile<ELFT>::getRelocationSymbol(DataRefImpl Rel) const {
  uint32_t symbolIdx;
  const Elf_Shdr *sec = getRelSection(Rel);
  if (sec->sh_type == ELF::SHT_REL)
    symbolIdx = getRel(Rel)->getSymbol(EF.isMips64EL());
  else
    symbolIdx = getRela(Rel)->getSymbol(EF.isMips64EL());
  if (!symbolIdx)
    return symbol_end();

  bool IsDyn = Rel.d.b & 1;
  DataRefImpl SymbolData;
  if (IsDyn)
    SymbolData = toDRI(EF.getDotDynSymSec(), symbolIdx);
  else
    SymbolData = toDRI(EF.getDotSymtabSec(), symbolIdx);
  return symbol_iterator(SymbolRef(SymbolData, this));
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getRelocationOffset(DataRefImpl Rel) const {
  assert(EF.getHeader()->e_type == ELF::ET_REL &&
         "Only relocatable object files have relocation offsets");
  const Elf_Shdr *sec = getRelSection(Rel);
  if (sec->sh_type == ELF::SHT_REL)
    return getRel(Rel)->r_offset;

  return getRela(Rel)->r_offset;
}

template <class ELFT>
uint64_t ELFObjectFile<ELFT>::getRelocationType(DataRefImpl Rel) const {
  const Elf_Shdr *sec = getRelSection(Rel);
  if (sec->sh_type == ELF::SHT_REL)
    return getRel(Rel)->getType(EF.isMips64EL());
  else
    return getRela(Rel)->getType(EF.isMips64EL());
}

template <class ELFT>
StringRef ELFObjectFile<ELFT>::getRelocationTypeName(uint32_t Type) const {
  return getELFRelocationTypeName(EF.getHeader()->e_machine, Type);
}

template <class ELFT>
void ELFObjectFile<ELFT>::getRelocationTypeName(
    DataRefImpl Rel, SmallVectorImpl<char> &Result) const {
  uint32_t type = getRelocationType(Rel);
  EF.getRelocationTypeName(type, Result);
}

template <class ELFT>
ErrorOr<int64_t>
ELFObjectFile<ELFT>::getRelocationAddend(DataRefImpl Rel) const {
  if (getRelSection(Rel)->sh_type != ELF::SHT_RELA)
    return object_error::parse_failed;
  return (int64_t)getRela(Rel)->r_addend;
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Sym *
ELFObjectFile<ELFT>::getSymbol(DataRefImpl Symb) const {
  return &*toELFSymIter(Symb);
}

template <class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Rel *
ELFObjectFile<ELFT>::getRel(DataRefImpl Rel) const {
  assert(getRelSection(Rel)->sh_type == ELF::SHT_REL);
  return EF.template getEntry<Elf_Rel>(Rel.d.a, Rel.d.b >> 1);
}

template <class ELFT>
const typename ELFObjectFile<ELFT>::Elf_Rela *
ELFObjectFile<ELFT>::getRela(DataRefImpl Rela) const {
  assert(getRelSection(Rela)->sh_type == ELF::SHT_RELA);
  return EF.template getEntry<Elf_Rela>(Rela.d.a, Rela.d.b >> 1);
}

template <class ELFT>
ELFObjectFile<ELFT>::ELFObjectFile(MemoryBufferRef Object, std::error_code &EC)
    : ELFObjectFileBase(
          getELFType(static_cast<endianness>(ELFT::TargetEndianness) ==
                         support::little,
                     ELFT::Is64Bits),
          Object),
      EF(Data.getBuffer(), EC) {}

template <class ELFT>
basic_symbol_iterator ELFObjectFile<ELFT>::symbol_begin_impl() const {
  DataRefImpl Sym = toDRI(EF.getDotSymtabSec(), 0);
  return basic_symbol_iterator(SymbolRef(Sym, this));
}

template <class ELFT>
basic_symbol_iterator ELFObjectFile<ELFT>::symbol_end_impl() const {
  const Elf_Shdr *SymTab = EF.getDotSymtabSec();
  if (!SymTab)
    return symbol_begin_impl();
  DataRefImpl Sym = toDRI(SymTab, SymTab->sh_size / sizeof(Elf_Sym));
  return basic_symbol_iterator(SymbolRef(Sym, this));
}

template <class ELFT>
elf_symbol_iterator ELFObjectFile<ELFT>::dynamic_symbol_begin() const {
  DataRefImpl Sym = toDRI(EF.getDotDynSymSec(), 0);
  return symbol_iterator(SymbolRef(Sym, this));
}

template <class ELFT>
elf_symbol_iterator ELFObjectFile<ELFT>::dynamic_symbol_end() const {
  const Elf_Shdr *SymTab = EF.getDotDynSymSec();
  DataRefImpl Sym = toDRI(SymTab, SymTab->sh_size / sizeof(Elf_Sym));
  return basic_symbol_iterator(SymbolRef(Sym, this));
}

template <class ELFT>
section_iterator ELFObjectFile<ELFT>::section_begin() const {
  return section_iterator(SectionRef(toDRI(EF.section_begin()), this));
}

template <class ELFT>
section_iterator ELFObjectFile<ELFT>::section_end() const {
  return section_iterator(SectionRef(toDRI(EF.section_end()), this));
}

template <class ELFT>
StringRef ELFObjectFile<ELFT>::getLoadName() const {
  Elf_Dyn_Iter DI = EF.dynamic_table_begin();
  Elf_Dyn_Iter DE = EF.dynamic_table_end();

  while (DI != DE && DI->getTag() != ELF::DT_SONAME)
    ++DI;

  if (DI != DE)
    return EF.getDynamicString(DI->getVal());
  return "";
}

template <class ELFT>
uint8_t ELFObjectFile<ELFT>::getBytesInAddress() const {
  return ELFT::Is64Bits ? 8 : 4;
}

template <class ELFT>
StringRef ELFObjectFile<ELFT>::getFileFormatName() const {
  bool IsLittleEndian = ELFT::TargetEndianness == support::little;
  switch (EF.getHeader()->e_ident[ELF::EI_CLASS]) {
  case ELF::ELFCLASS32:
    switch (EF.getHeader()->e_machine) {
    case ELF::EM_386:
      return "ELF32-i386";
    case ELF::EM_X86_64:
      return "ELF32-x86-64";
    case ELF::EM_ARM:
      return (IsLittleEndian ? "ELF32-arm-little" : "ELF32-arm-big");
    case ELF::EM_HEXAGON:
      return "ELF32-hexagon";
    case ELF::EM_MIPS:
      return "ELF32-mips";
    case ELF::EM_PPC:
      return "ELF32-ppc";
    case ELF::EM_SPARC:
    case ELF::EM_SPARC32PLUS:
      return "ELF32-sparc";
    default:
      return "ELF32-unknown";
    }
  case ELF::ELFCLASS64:
    switch (EF.getHeader()->e_machine) {
    case ELF::EM_386:
      return "ELF64-i386";
    case ELF::EM_X86_64:
      return "ELF64-x86-64";
    case ELF::EM_AARCH64:
      return (IsLittleEndian ? "ELF64-aarch64-little" : "ELF64-aarch64-big");
    case ELF::EM_PPC64:
      return "ELF64-ppc64";
    case ELF::EM_S390:
      return "ELF64-s390";
    case ELF::EM_SPARCV9:
      return "ELF64-sparc";
    case ELF::EM_MIPS:
      return "ELF64-mips";
    default:
      return "ELF64-unknown";
    }
  default:
    // FIXME: Proper error handling.
    report_fatal_error("Invalid ELFCLASS!");
  }
}

template <class ELFT>
unsigned ELFObjectFile<ELFT>::getArch() const {
  bool IsLittleEndian = ELFT::TargetEndianness == support::little;
  switch (EF.getHeader()->e_machine) {
  case ELF::EM_386:
    return Triple::x86;
  case ELF::EM_X86_64:
    return Triple::x86_64;
  case ELF::EM_AARCH64:
    return Triple::aarch64;
  case ELF::EM_ARM:
    return Triple::arm;
  case ELF::EM_HEXAGON:
    return Triple::hexagon;
  case ELF::EM_MIPS:
    switch (EF.getHeader()->e_ident[ELF::EI_CLASS]) {
    case ELF::ELFCLASS32:
      return IsLittleEndian ? Triple::mipsel : Triple::mips;
    case ELF::ELFCLASS64:
      return IsLittleEndian ? Triple::mips64el : Triple::mips64;
    default:
      report_fatal_error("Invalid ELFCLASS!");
    }
  case ELF::EM_PPC:
    return Triple::ppc;
  case ELF::EM_PPC64:
    return IsLittleEndian ? Triple::ppc64le : Triple::ppc64;
  case ELF::EM_S390:
    return Triple::systemz;

  case ELF::EM_SPARC:
  case ELF::EM_SPARC32PLUS:
    return IsLittleEndian ? Triple::sparcel : Triple::sparc;
  case ELF::EM_SPARCV9:
    return Triple::sparcv9;

  default:
    return Triple::UnknownArch;
  }
}

template <class ELFT>
ELFObjectFileBase::elf_symbol_iterator_range
ELFObjectFile<ELFT>::getDynamicSymbolIterators() const {
  return make_range(dynamic_symbol_begin(), dynamic_symbol_end());
}

template <class ELFT> bool ELFObjectFile<ELFT>::isRelocatableObject() const {
  return EF.getHeader()->e_type == ELF::ET_REL;
}

}
}

#endif
