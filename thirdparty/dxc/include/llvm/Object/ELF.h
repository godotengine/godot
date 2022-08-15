//===- ELF.h - ELF object file implementation -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the ELFFile template class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELF_H
#define LLVM_OBJECT_ELF_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <limits>
#include <utility>

namespace llvm {
namespace object {

StringRef getELFRelocationTypeName(uint32_t Machine, uint32_t Type);

// Subclasses of ELFFile may need this for template instantiation
inline std::pair<unsigned char, unsigned char>
getElfArchType(StringRef Object) {
  if (Object.size() < ELF::EI_NIDENT)
    return std::make_pair((uint8_t)ELF::ELFCLASSNONE,
                          (uint8_t)ELF::ELFDATANONE);
  return std::make_pair((uint8_t)Object[ELF::EI_CLASS],
                        (uint8_t)Object[ELF::EI_DATA]);
}

template <class ELFT>
class ELFFile {
public:
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  typedef typename std::conditional<ELFT::Is64Bits,
                                    uint64_t, uint32_t>::type uintX_t;

  /// \brief Iterate over constant sized entities.
  template <class EntT>
  class ELFEntityIterator {
  public:
    typedef ptrdiff_t difference_type;
    typedef EntT value_type;
    typedef std::forward_iterator_tag iterator_category;
    typedef value_type &reference;
    typedef value_type *pointer;

    /// \brief Default construct iterator.
    ELFEntityIterator() : EntitySize(0), Current(nullptr) {}
    ELFEntityIterator(uintX_t EntSize, const char *Start)
        : EntitySize(EntSize), Current(Start) {}

    reference operator *() {
      assert(Current && "Attempted to dereference an invalid iterator!");
      return *reinterpret_cast<pointer>(Current);
    }

    pointer operator ->() {
      assert(Current && "Attempted to dereference an invalid iterator!");
      return reinterpret_cast<pointer>(Current);
    }

    bool operator ==(const ELFEntityIterator &Other) {
      return Current == Other.Current;
    }

    bool operator !=(const ELFEntityIterator &Other) {
      return !(*this == Other);
    }

    ELFEntityIterator &operator ++() {
      assert(Current && "Attempted to increment an invalid iterator!");
      Current += EntitySize;
      return *this;
    }

    ELFEntityIterator &operator+(difference_type n) {
      assert(Current && "Attempted to increment an invalid iterator!");
      Current += (n * EntitySize);
      return *this;
    }

    ELFEntityIterator &operator-(difference_type n) {
      assert(Current && "Attempted to subtract an invalid iterator!");
      Current -= (n * EntitySize);
      return *this;
    }

    ELFEntityIterator operator ++(int) {
      ELFEntityIterator Tmp = *this;
      ++*this;
      return Tmp;
    }

    difference_type operator -(const ELFEntityIterator &Other) const {
      assert(EntitySize == Other.EntitySize &&
             "Subtracting iterators of different EntitySize!");
      return (Current - Other.Current) / EntitySize;
    }

    const char *get() const { return Current; }

    uintX_t getEntSize() const { return EntitySize; }

  private:
    uintX_t EntitySize;
    const char *Current;
  };

  typedef Elf_Ehdr_Impl<ELFT> Elf_Ehdr;
  typedef Elf_Shdr_Impl<ELFT> Elf_Shdr;
  typedef Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef Elf_Dyn_Impl<ELFT> Elf_Dyn;
  typedef Elf_Phdr_Impl<ELFT> Elf_Phdr;
  typedef Elf_Rel_Impl<ELFT, false> Elf_Rel;
  typedef Elf_Rel_Impl<ELFT, true> Elf_Rela;
  typedef Elf_Verdef_Impl<ELFT> Elf_Verdef;
  typedef Elf_Verdaux_Impl<ELFT> Elf_Verdaux;
  typedef Elf_Verneed_Impl<ELFT> Elf_Verneed;
  typedef Elf_Vernaux_Impl<ELFT> Elf_Vernaux;
  typedef Elf_Versym_Impl<ELFT> Elf_Versym;
  typedef Elf_Hash_Impl<ELFT> Elf_Hash;
  typedef ELFEntityIterator<const Elf_Dyn> Elf_Dyn_Iter;
  typedef iterator_range<Elf_Dyn_Iter> Elf_Dyn_Range;
  typedef ELFEntityIterator<const Elf_Rela> Elf_Rela_Iter;
  typedef ELFEntityIterator<const Elf_Rel> Elf_Rel_Iter;
  typedef iterator_range<const Elf_Shdr *> Elf_Shdr_Range;

  /// \brief Archive files are 2 byte aligned, so we need this for
  ///     PointerIntPair to work.
  template <typename T>
  class ArchivePointerTypeTraits {
  public:
    static inline const void *getAsVoidPointer(T *P) { return P; }
    static inline T *getFromVoidPointer(const void *P) {
      return static_cast<T *>(P);
    }
    enum { NumLowBitsAvailable = 1 };
  };

  typedef iterator_range<const Elf_Sym *> Elf_Sym_Range;

private:
  typedef SmallVector<const Elf_Shdr *, 2> Sections_t;
  typedef DenseMap<unsigned, unsigned> IndexMap_t;

  StringRef Buf;

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Buf.data());
  }

  const Elf_Ehdr *Header;
  const Elf_Shdr *SectionHeaderTable = nullptr;
  StringRef DotShstrtab;                    // Section header string table.
  StringRef DotStrtab;                      // Symbol header string table.
  const Elf_Shdr *dot_symtab_sec = nullptr; // Symbol table section.
  const Elf_Shdr *DotDynSymSec = nullptr;   // Dynamic symbol table section.
  const Elf_Hash *HashTable = nullptr;

  const Elf_Shdr *SymbolTableSectionHeaderIndex = nullptr;
  DenseMap<const Elf_Sym *, ELF::Elf64_Word> ExtendedSymbolTable;

  const Elf_Shdr *dot_gnu_version_sec = nullptr;   // .gnu.version
  const Elf_Shdr *dot_gnu_version_r_sec = nullptr; // .gnu.version_r
  const Elf_Shdr *dot_gnu_version_d_sec = nullptr; // .gnu.version_d

  /// \brief Represents a region described by entries in the .dynamic table.
  struct DynRegionInfo {
    DynRegionInfo() : Addr(nullptr), Size(0), EntSize(0) {}
    /// \brief Address in current address space.
    const void *Addr;
    /// \brief Size in bytes of the region.
    uintX_t Size;
    /// \brief Size of each entity in the region.
    uintX_t EntSize;
  };

  DynRegionInfo DynamicRegion;
  DynRegionInfo DynHashRegion;
  DynRegionInfo DynStrRegion;
  DynRegionInfo DynRelaRegion;

  // Pointer to SONAME entry in dynamic string table
  // This is set the first time getLoadName is called.
  mutable const char *dt_soname = nullptr;

  // Records for each version index the corresponding Verdef or Vernaux entry.
  // This is filled the first time LoadVersionMap() is called.
  class VersionMapEntry : public PointerIntPair<const void*, 1> {
    public:
    // If the integer is 0, this is an Elf_Verdef*.
    // If the integer is 1, this is an Elf_Vernaux*.
    VersionMapEntry() : PointerIntPair<const void*, 1>(nullptr, 0) { }
    VersionMapEntry(const Elf_Verdef *verdef)
        : PointerIntPair<const void*, 1>(verdef, 0) { }
    VersionMapEntry(const Elf_Vernaux *vernaux)
        : PointerIntPair<const void*, 1>(vernaux, 1) { }
    bool isNull() const { return getPointer() == nullptr; }
    bool isVerdef() const { return !isNull() && getInt() == 0; }
    bool isVernaux() const { return !isNull() && getInt() == 1; }
    const Elf_Verdef *getVerdef() const {
      return isVerdef() ? (const Elf_Verdef*)getPointer() : nullptr;
    }
    const Elf_Vernaux *getVernaux() const {
      return isVernaux() ? (const Elf_Vernaux*)getPointer() : nullptr;
    }
  };
  mutable SmallVector<VersionMapEntry, 16> VersionMap;
  void LoadVersionDefs(const Elf_Shdr *sec) const;
  void LoadVersionNeeds(const Elf_Shdr *ec) const;
  void LoadVersionMap() const;

  void scanDynamicTable();

public:
  template<typename T>
  const T        *getEntry(uint32_t Section, uint32_t Entry) const;
  template <typename T>
  const T *getEntry(const Elf_Shdr *Section, uint32_t Entry) const;

  const Elf_Shdr *getDotSymtabSec() const { return dot_symtab_sec; }
  const Elf_Shdr *getDotDynSymSec() const { return DotDynSymSec; }
  const Elf_Hash *getHashTable() const { return HashTable; }

  ErrorOr<StringRef> getStringTable(const Elf_Shdr *Section) const;
  const char *getDynamicString(uintX_t Offset) const;
  ErrorOr<StringRef> getSymbolVersion(const Elf_Shdr *section,
                                      const Elf_Sym *Symb,
                                      bool &IsDefault) const;
  void VerifyStrTab(const Elf_Shdr *sh) const;

  StringRef getRelocationTypeName(uint32_t Type) const;
  void getRelocationTypeName(uint32_t Type,
                             SmallVectorImpl<char> &Result) const;

  /// \brief Get the symbol table section and symbol for a given relocation.
  template <class RelT>
  std::pair<const Elf_Shdr *, const Elf_Sym *>
  getRelocationSymbol(const Elf_Shdr *RelSec, const RelT *Rel) const;

  ELFFile(StringRef Object, std::error_code &EC);

  bool isMipsELF64() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64;
  }

  bool isMips64EL() const {
    return Header->e_machine == ELF::EM_MIPS &&
      Header->getFileClass() == ELF::ELFCLASS64 &&
      Header->getDataEncoding() == ELF::ELFDATA2LSB;
  }

  const Elf_Shdr *section_begin() const;
  const Elf_Shdr *section_end() const;
  Elf_Shdr_Range sections() const {
    return make_range(section_begin(), section_end());
  }

  const Elf_Sym *symbol_begin() const;
  const Elf_Sym *symbol_end() const;
  Elf_Sym_Range symbols() const {
    return make_range(symbol_begin(), symbol_end());
  }

  Elf_Dyn_Iter dynamic_table_begin() const;
  /// \param NULLEnd use one past the first DT_NULL entry as the end instead of
  /// the section size.
  Elf_Dyn_Iter dynamic_table_end(bool NULLEnd = false) const;
  Elf_Dyn_Range dynamic_table(bool NULLEnd = false) const {
    return make_range(dynamic_table_begin(), dynamic_table_end(NULLEnd));
  }

  const Elf_Sym *dynamic_symbol_begin() const {
    if (!DotDynSymSec)
      return nullptr;
    if (DotDynSymSec->sh_entsize != sizeof(Elf_Sym))
      report_fatal_error("Invalid symbol size");
    return reinterpret_cast<const Elf_Sym *>(base() + DotDynSymSec->sh_offset);
  }

  const Elf_Sym *dynamic_symbol_end() const {
    if (!DotDynSymSec)
      return nullptr;
    return reinterpret_cast<const Elf_Sym *>(base() + DotDynSymSec->sh_offset +
                                             DotDynSymSec->sh_size);
  }

  Elf_Sym_Range dynamic_symbols() const {
    return make_range(dynamic_symbol_begin(), dynamic_symbol_end());
  }

  Elf_Rela_Iter dyn_rela_begin() const {
    if (DynRelaRegion.Addr)
      return Elf_Rela_Iter(DynRelaRegion.EntSize,
        (const char *)DynRelaRegion.Addr);
    return Elf_Rela_Iter(0, nullptr);
  }

  Elf_Rela_Iter dyn_rela_end() const {
    if (DynRelaRegion.Addr)
      return Elf_Rela_Iter(
        DynRelaRegion.EntSize,
        (const char *)DynRelaRegion.Addr + DynRelaRegion.Size);
    return Elf_Rela_Iter(0, nullptr);
  }

  Elf_Rela_Iter rela_begin(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(sec->sh_entsize,
                         (const char *)(base() + sec->sh_offset));
  }

  Elf_Rela_Iter rela_end(const Elf_Shdr *sec) const {
    return Elf_Rela_Iter(
        sec->sh_entsize,
        (const char *)(base() + sec->sh_offset + sec->sh_size));
  }

  Elf_Rel_Iter rel_begin(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec->sh_entsize,
                        (const char *)(base() + sec->sh_offset));
  }

  Elf_Rel_Iter rel_end(const Elf_Shdr *sec) const {
    return Elf_Rel_Iter(sec->sh_entsize,
                        (const char *)(base() + sec->sh_offset + sec->sh_size));
  }

  /// \brief Iterate over program header table.
  typedef ELFEntityIterator<const Elf_Phdr> Elf_Phdr_Iter;

  Elf_Phdr_Iter program_header_begin() const {
    return Elf_Phdr_Iter(Header->e_phentsize,
                         (const char*)base() + Header->e_phoff);
  }

  Elf_Phdr_Iter program_header_end() const {
    return Elf_Phdr_Iter(Header->e_phentsize,
                         (const char*)base() +
                           Header->e_phoff +
                           (Header->e_phnum * Header->e_phentsize));
  }

  uint64_t getNumSections() const;
  uintX_t getStringTableIndex() const;
  ELF::Elf64_Word getExtendedSymbolTableIndex(const Elf_Sym *symb) const;
  const Elf_Ehdr *getHeader() const { return Header; }
  ErrorOr<const Elf_Shdr *> getSection(const Elf_Sym *symb) const;
  ErrorOr<const Elf_Shdr *> getSection(uint32_t Index) const;
  const Elf_Sym *getSymbol(uint32_t index) const;

  ErrorOr<StringRef> getStaticSymbolName(const Elf_Sym *Symb) const;
  ErrorOr<StringRef> getDynamicSymbolName(const Elf_Sym *Symb) const;
  ErrorOr<StringRef> getSymbolName(const Elf_Sym *Symb, bool IsDynamic) const;

  ErrorOr<StringRef> getSectionName(const Elf_Shdr *Section) const;
  ErrorOr<ArrayRef<uint8_t> > getSectionContents(const Elf_Shdr *Sec) const;
  StringRef getLoadName() const;
};

typedef ELFFile<ELFType<support::little, false>> ELF32LEFile;
typedef ELFFile<ELFType<support::little, true>> ELF64LEFile;
typedef ELFFile<ELFType<support::big, false>> ELF32BEFile;
typedef ELFFile<ELFType<support::big, true>> ELF64BEFile;

// Iterate through the version definitions, and place each Elf_Verdef
// in the VersionMap according to its index.
template <class ELFT>
void ELFFile<ELFT>::LoadVersionDefs(const Elf_Shdr *sec) const {
  unsigned vd_size = sec->sh_size;  // Size of section in bytes
  unsigned vd_count = sec->sh_info; // Number of Verdef entries
  const char *sec_start = (const char*)base() + sec->sh_offset;
  const char *sec_end = sec_start + vd_size;
  // The first Verdef entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vd_count; i++) {
    if (p + sizeof(Elf_Verdef) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version definitions.");
    const Elf_Verdef *vd = reinterpret_cast<const Elf_Verdef *>(p);
    if (vd->vd_version != ELF::VER_DEF_CURRENT)
      report_fatal_error("Unexpected verdef version");
    size_t index = vd->vd_ndx & ELF::VERSYM_VERSION;
    if (index >= VersionMap.size())
      VersionMap.resize(index + 1);
    VersionMap[index] = VersionMapEntry(vd);
    p += vd->vd_next;
  }
}

// Iterate through the versions needed section, and place each Elf_Vernaux
// in the VersionMap according to its index.
template <class ELFT>
void ELFFile<ELFT>::LoadVersionNeeds(const Elf_Shdr *sec) const {
  unsigned vn_size = sec->sh_size;  // Size of section in bytes
  unsigned vn_count = sec->sh_info; // Number of Verneed entries
  const char *sec_start = (const char *)base() + sec->sh_offset;
  const char *sec_end = sec_start + vn_size;
  // The first Verneed entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vn_count; i++) {
    if (p + sizeof(Elf_Verneed) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version needed records.");
    const Elf_Verneed *vn = reinterpret_cast<const Elf_Verneed *>(p);
    if (vn->vn_version != ELF::VER_NEED_CURRENT)
      report_fatal_error("Unexpected verneed version");
    // Iterate through the Vernaux entries
    const char *paux = p + vn->vn_aux;
    for (unsigned j = 0; j < vn->vn_cnt; j++) {
      if (paux + sizeof(Elf_Vernaux) > sec_end)
        report_fatal_error("Section ended unexpected while scanning auxiliary "
                           "version needed records.");
      const Elf_Vernaux *vna = reinterpret_cast<const Elf_Vernaux *>(paux);
      size_t index = vna->vna_other & ELF::VERSYM_VERSION;
      if (index >= VersionMap.size())
        VersionMap.resize(index + 1);
      VersionMap[index] = VersionMapEntry(vna);
      paux += vna->vna_next;
    }
    p += vn->vn_next;
  }
}

template <class ELFT>
void ELFFile<ELFT>::LoadVersionMap() const {
  // If there is no dynamic symtab or version table, there is nothing to do.
  if (!DotDynSymSec || !dot_gnu_version_sec)
    return;

  // Has the VersionMap already been loaded?
  if (VersionMap.size() > 0)
    return;

  // The first two version indexes are reserved.
  // Index 0 is LOCAL, index 1 is GLOBAL.
  VersionMap.push_back(VersionMapEntry());
  VersionMap.push_back(VersionMapEntry());

  if (dot_gnu_version_d_sec)
    LoadVersionDefs(dot_gnu_version_d_sec);

  if (dot_gnu_version_r_sec)
    LoadVersionNeeds(dot_gnu_version_r_sec);
}

template <class ELFT>
ELF::Elf64_Word
ELFFile<ELFT>::getExtendedSymbolTableIndex(const Elf_Sym *symb) const {
  assert(symb->st_shndx == ELF::SHN_XINDEX);
  return ExtendedSymbolTable.lookup(symb);
}

template <class ELFT>
ErrorOr<const typename ELFFile<ELFT>::Elf_Shdr *>
ELFFile<ELFT>::getSection(const Elf_Sym *symb) const {
  uint32_t Index = symb->st_shndx;
  if (Index == ELF::SHN_XINDEX)
    return getSection(ExtendedSymbolTable.lookup(symb));
  if (Index == ELF::SHN_UNDEF || Index >= ELF::SHN_LORESERVE)
    return nullptr;
  return getSection(symb->st_shndx);
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Sym *
ELFFile<ELFT>::getSymbol(uint32_t Index) const {
  return &*(symbol_begin() + Index);
}

template <class ELFT>
ErrorOr<ArrayRef<uint8_t> >
ELFFile<ELFT>::getSectionContents(const Elf_Shdr *Sec) const {
  if (Sec->sh_offset + Sec->sh_size > Buf.size())
    return object_error::parse_failed;
  const uint8_t *Start = base() + Sec->sh_offset;
  return makeArrayRef(Start, Sec->sh_size);
}

template <class ELFT>
StringRef ELFFile<ELFT>::getRelocationTypeName(uint32_t Type) const {
  return getELFRelocationTypeName(Header->e_machine, Type);
}

template <class ELFT>
void ELFFile<ELFT>::getRelocationTypeName(uint32_t Type,
                                          SmallVectorImpl<char> &Result) const {
  if (!isMipsELF64()) {
    StringRef Name = getRelocationTypeName(Type);
    Result.append(Name.begin(), Name.end());
  } else {
    // The Mips N64 ABI allows up to three operations to be specified per
    // relocation record. Unfortunately there's no easy way to test for the
    // presence of N64 ELFs as they have no special flag that identifies them
    // as being N64. We can safely assume at the moment that all Mips
    // ELFCLASS64 ELFs are N64. New Mips64 ABIs should provide enough
    // information to disambiguate between old vs new ABIs.
    uint8_t Type1 = (Type >> 0) & 0xFF;
    uint8_t Type2 = (Type >> 8) & 0xFF;
    uint8_t Type3 = (Type >> 16) & 0xFF;

    // Concat all three relocation type names.
    StringRef Name = getRelocationTypeName(Type1);
    Result.append(Name.begin(), Name.end());

    Name = getRelocationTypeName(Type2);
    Result.append(1, '/');
    Result.append(Name.begin(), Name.end());

    Name = getRelocationTypeName(Type3);
    Result.append(1, '/');
    Result.append(Name.begin(), Name.end());
  }
}

template <class ELFT>
template <class RelT>
std::pair<const typename ELFFile<ELFT>::Elf_Shdr *,
          const typename ELFFile<ELFT>::Elf_Sym *>
ELFFile<ELFT>::getRelocationSymbol(const Elf_Shdr *Sec, const RelT *Rel) const {
  if (!Sec->sh_link)
    return std::make_pair(nullptr, nullptr);
  ErrorOr<const Elf_Shdr *> SymTableOrErr = getSection(Sec->sh_link);
  if (std::error_code EC = SymTableOrErr.getError())
    report_fatal_error(EC.message());
  const Elf_Shdr *SymTable = *SymTableOrErr;
  return std::make_pair(
      SymTable, getEntry<Elf_Sym>(SymTable, Rel->getSymbol(isMips64EL())));
}

template <class ELFT>
uint64_t ELFFile<ELFT>::getNumSections() const {
  assert(Header && "Header not initialized!");
  if (Header->e_shnum == ELF::SHN_UNDEF && Header->e_shoff > 0) {
    assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
    return SectionHeaderTable->sh_size;
  }
  return Header->e_shnum;
}

template <class ELFT>
typename ELFFile<ELFT>::uintX_t ELFFile<ELFT>::getStringTableIndex() const {
  if (Header->e_shnum == ELF::SHN_UNDEF) {
    if (Header->e_shstrndx == ELF::SHN_HIRESERVE)
      return SectionHeaderTable->sh_link;
    if (Header->e_shstrndx >= getNumSections())
      return 0;
  }
  return Header->e_shstrndx;
}

template <class ELFT>
ELFFile<ELFT>::ELFFile(StringRef Object, std::error_code &EC)
    : Buf(Object) {
  const uint64_t FileSize = Buf.size();

  if (sizeof(Elf_Ehdr) > FileSize) {
    // File too short!
    EC = object_error::parse_failed;
    return;
  }

  Header = reinterpret_cast<const Elf_Ehdr *>(base());

  if (Header->e_shoff == 0) {
    scanDynamicTable();
    return;
  }

  const uint64_t SectionTableOffset = Header->e_shoff;

  if (SectionTableOffset + sizeof(Elf_Shdr) > FileSize) {
    // Section header table goes past end of file!
    EC = object_error::parse_failed;
    return;
  }

  // The getNumSections() call below depends on SectionHeaderTable being set.
  SectionHeaderTable =
    reinterpret_cast<const Elf_Shdr *>(base() + SectionTableOffset);
  const uint64_t SectionTableSize = getNumSections() * Header->e_shentsize;

  if (SectionTableOffset + SectionTableSize > FileSize) {
    // Section table goes past end of file!
    EC = object_error::parse_failed;
    return;
  }

  // Scan sections for special sections.

  for (const Elf_Shdr &Sec : sections()) {
    switch (Sec.sh_type) {
    case ELF::SHT_HASH:
      if (HashTable) {
        EC = object_error::parse_failed;
        return;
      }
      HashTable = reinterpret_cast<const Elf_Hash *>(base() + Sec.sh_offset);
      break;
    case ELF::SHT_SYMTAB_SHNDX:
      if (SymbolTableSectionHeaderIndex) {
        // More than one .symtab_shndx!
        EC = object_error::parse_failed;
        return;
      }
      SymbolTableSectionHeaderIndex = &Sec;
      break;
    case ELF::SHT_SYMTAB: {
      if (dot_symtab_sec) {
        // More than one .symtab!
        EC = object_error::parse_failed;
        return;
      }
      dot_symtab_sec = &Sec;
      ErrorOr<const Elf_Shdr *> SectionOrErr = getSection(Sec.sh_link);
      if ((EC = SectionOrErr.getError()))
        return;
      ErrorOr<StringRef> SymtabOrErr = getStringTable(*SectionOrErr);
      if ((EC = SymtabOrErr.getError()))
        return;
      DotStrtab = *SymtabOrErr;
    } break;
    case ELF::SHT_DYNSYM: {
      if (DotDynSymSec) {
        // More than one .dynsym!
        EC = object_error::parse_failed;
        return;
      }
      DotDynSymSec = &Sec;
      ErrorOr<const Elf_Shdr *> SectionOrErr = getSection(Sec.sh_link);
      if ((EC = SectionOrErr.getError()))
        return;
      ErrorOr<StringRef> SymtabOrErr = getStringTable(*SectionOrErr);
      if ((EC = SymtabOrErr.getError()))
        return;
      DynStrRegion.Addr = SymtabOrErr->data();
      DynStrRegion.Size = SymtabOrErr->size();
      DynStrRegion.EntSize = 1;
      break;
    }
    case ELF::SHT_DYNAMIC:
      if (DynamicRegion.Addr) {
        // More than one .dynamic!
        EC = object_error::parse_failed;
        return;
      }
      DynamicRegion.Addr = base() + Sec.sh_offset;
      DynamicRegion.Size = Sec.sh_size;
      DynamicRegion.EntSize = Sec.sh_entsize;
      break;
    case ELF::SHT_GNU_versym:
      if (dot_gnu_version_sec != nullptr) {
        // More than one .gnu.version section!
        EC = object_error::parse_failed;
        return;
      }
      dot_gnu_version_sec = &Sec;
      break;
    case ELF::SHT_GNU_verdef:
      if (dot_gnu_version_d_sec != nullptr) {
        // More than one .gnu.version_d section!
        EC = object_error::parse_failed;
        return;
      }
      dot_gnu_version_d_sec = &Sec;
      break;
    case ELF::SHT_GNU_verneed:
      if (dot_gnu_version_r_sec != nullptr) {
        // More than one .gnu.version_r section!
        EC = object_error::parse_failed;
        return;
      }
      dot_gnu_version_r_sec = &Sec;
      break;
    }
  }

  // Get string table sections.
  ErrorOr<const Elf_Shdr *> StrTabSecOrErr = getSection(getStringTableIndex());
  if ((EC = StrTabSecOrErr.getError()))
    return;

  ErrorOr<StringRef> SymtabOrErr = getStringTable(*StrTabSecOrErr);
  if ((EC = SymtabOrErr.getError()))
    return;
  DotShstrtab = *SymtabOrErr;

  // Build symbol name side-mapping if there is one.
  if (SymbolTableSectionHeaderIndex) {
    const Elf_Word *ShndxTable = reinterpret_cast<const Elf_Word*>(base() +
                                      SymbolTableSectionHeaderIndex->sh_offset);
    for (const Elf_Sym &S : symbols()) {
      if (*ShndxTable != ELF::SHN_UNDEF)
        ExtendedSymbolTable[&S] = *ShndxTable;
      ++ShndxTable;
    }
  }

  scanDynamicTable();

  EC = std::error_code();
}

template <class ELFT>
void ELFFile<ELFT>::scanDynamicTable() {
  // Build load-address to file-offset map.
  typedef IntervalMap<
      uintX_t, uintptr_t,
      IntervalMapImpl::NodeSizer<uintX_t, uintptr_t>::LeafSize,
      IntervalMapHalfOpenInfo<uintX_t>> LoadMapT;
  typename LoadMapT::Allocator Alloc;
  // Allocate the IntervalMap on the heap to work around MSVC bug where the
  // stack doesn't get realigned despite LoadMap having alignment 8 (PR24113).
  std::unique_ptr<LoadMapT> LoadMap(new LoadMapT(Alloc));

  for (Elf_Phdr_Iter PhdrI = program_header_begin(),
                     PhdrE = program_header_end();
       PhdrI != PhdrE; ++PhdrI) {
    if (PhdrI->p_type == ELF::PT_DYNAMIC) {
      DynamicRegion.Addr = base() + PhdrI->p_offset;
      DynamicRegion.Size = PhdrI->p_filesz;
      DynamicRegion.EntSize = sizeof(Elf_Dyn);
      continue;
    }
    if (PhdrI->p_type != ELF::PT_LOAD)
      continue;
    if (PhdrI->p_filesz == 0)
      continue;
    LoadMap->insert(PhdrI->p_vaddr, PhdrI->p_vaddr + PhdrI->p_filesz,
                    PhdrI->p_offset);
  }

  auto toMappedAddr = [&](uint64_t VAddr) -> const uint8_t * {
    auto I = LoadMap->find(VAddr);
    if (I == LoadMap->end())
      return nullptr;
    return this->base() + I.value() + (VAddr - I.start());
  };

  for (Elf_Dyn_Iter DynI = dynamic_table_begin(), DynE = dynamic_table_end();
       DynI != DynE; ++DynI) {
    switch (DynI->d_tag) {
    case ELF::DT_HASH:
      if (HashTable)
        continue;
      HashTable =
          reinterpret_cast<const Elf_Hash *>(toMappedAddr(DynI->getPtr()));
      break;
    case ELF::DT_STRTAB:
      if (!DynStrRegion.Addr)
        DynStrRegion.Addr = toMappedAddr(DynI->getPtr());
      break;
    case ELF::DT_STRSZ:
      if (!DynStrRegion.Size)
        DynStrRegion.Size = DynI->getVal();
      break;
    case ELF::DT_RELA:
      if (!DynRelaRegion.Addr)
        DynRelaRegion.Addr = toMappedAddr(DynI->getPtr());
      break;
    case ELF::DT_RELASZ:
      DynRelaRegion.Size = DynI->getVal();
      break;
    case ELF::DT_RELAENT:
      DynRelaRegion.EntSize = DynI->getVal();
    }
  }
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *ELFFile<ELFT>::section_begin() const {
  if (Header->e_shentsize != sizeof(Elf_Shdr))
    report_fatal_error(
        "Invalid section header entry size (e_shentsize) in ELF header");
  return reinterpret_cast<const Elf_Shdr *>(base() + Header->e_shoff);
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Shdr *ELFFile<ELFT>::section_end() const {
  return section_begin() + getNumSections();
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Sym *ELFFile<ELFT>::symbol_begin() const {
  if (!dot_symtab_sec)
    return nullptr;
  if (dot_symtab_sec->sh_entsize != sizeof(Elf_Sym))
    report_fatal_error("Invalid symbol size");
  return reinterpret_cast<const Elf_Sym *>(base() + dot_symtab_sec->sh_offset);
}

template <class ELFT>
const typename ELFFile<ELFT>::Elf_Sym *ELFFile<ELFT>::symbol_end() const {
  if (!dot_symtab_sec)
    return nullptr;
  return reinterpret_cast<const Elf_Sym *>(base() + dot_symtab_sec->sh_offset +
                                           dot_symtab_sec->sh_size);
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Dyn_Iter
ELFFile<ELFT>::dynamic_table_begin() const {
  if (DynamicRegion.Addr)
    return Elf_Dyn_Iter(DynamicRegion.EntSize,
                        (const char *)DynamicRegion.Addr);
  return Elf_Dyn_Iter(0, nullptr);
}

template <class ELFT>
typename ELFFile<ELFT>::Elf_Dyn_Iter
ELFFile<ELFT>::dynamic_table_end(bool NULLEnd) const {
  if (!DynamicRegion.Addr)
    return Elf_Dyn_Iter(0, nullptr);
  Elf_Dyn_Iter Ret(DynamicRegion.EntSize,
                    (const char *)DynamicRegion.Addr + DynamicRegion.Size);

  if (NULLEnd) {
    Elf_Dyn_Iter Start = dynamic_table_begin();
    while (Start != Ret && Start->getTag() != ELF::DT_NULL)
      ++Start;

    // Include the DT_NULL.
    if (Start != Ret)
      ++Start;
    Ret = Start;
  }
  return Ret;
}

template <class ELFT>
StringRef ELFFile<ELFT>::getLoadName() const {
  if (!dt_soname) {
    dt_soname = "";
    // Find the DT_SONAME entry
    for (const auto &Entry : dynamic_table())
      if (Entry.getTag() == ELF::DT_SONAME) {
        dt_soname = getDynamicString(Entry.getVal());
        break;
      }
  }
  return dt_soname;
}

template <class ELFT>
template <typename T>
const T *ELFFile<ELFT>::getEntry(uint32_t Section, uint32_t Entry) const {
  ErrorOr<const Elf_Shdr *> Sec = getSection(Section);
  if (std::error_code EC = Sec.getError())
    report_fatal_error(EC.message());
  return getEntry<T>(*Sec, Entry);
}

template <class ELFT>
template <typename T>
const T *ELFFile<ELFT>::getEntry(const Elf_Shdr *Section,
                                 uint32_t Entry) const {
  return reinterpret_cast<const T *>(base() + Section->sh_offset +
                                     (Entry * Section->sh_entsize));
}

template <class ELFT>
ErrorOr<const typename ELFFile<ELFT>::Elf_Shdr *>
ELFFile<ELFT>::getSection(uint32_t Index) const {
  assert(SectionHeaderTable && "SectionHeaderTable not initialized!");
  if (Index >= getNumSections())
    return object_error::invalid_section_index;

  return reinterpret_cast<const Elf_Shdr *>(
      reinterpret_cast<const char *>(SectionHeaderTable) +
      (Index * Header->e_shentsize));
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getStringTable(const Elf_Shdr *Section) const {
  if (Section->sh_type != ELF::SHT_STRTAB)
    return object_error::parse_failed;
  uint64_t Offset = Section->sh_offset;
  uint64_t Size = Section->sh_size;
  if (Offset + Size > Buf.size())
    return object_error::parse_failed;
  StringRef Data((const char *)base() + Section->sh_offset, Size);
  if (Data[Size - 1] != '\0')
    return object_error::string_table_non_null_end;
  return Data;
}

template <class ELFT>
const char *ELFFile<ELFT>::getDynamicString(uintX_t Offset) const {
  if (Offset >= DynStrRegion.Size)
    return nullptr;
  return (const char *)DynStrRegion.Addr + Offset;
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getStaticSymbolName(const Elf_Sym *Symb) const {
  return Symb->getName(DotStrtab);
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getDynamicSymbolName(const Elf_Sym *Symb) const {
  return StringRef(getDynamicString(Symb->st_name));
}

template <class ELFT>
ErrorOr<StringRef> ELFFile<ELFT>::getSymbolName(const Elf_Sym *Symb,
                                                bool IsDynamic) const {
  if (IsDynamic)
    return getDynamicSymbolName(Symb);
  return getStaticSymbolName(Symb);
}

template <class ELFT>
ErrorOr<StringRef>
ELFFile<ELFT>::getSectionName(const Elf_Shdr *Section) const {
  uint32_t Offset = Section->sh_name;
  if (Offset >= DotShstrtab.size())
    return object_error::parse_failed;
  return StringRef(DotShstrtab.data() + Offset);
}

template <class ELFT>
ErrorOr<StringRef> ELFFile<ELFT>::getSymbolVersion(const Elf_Shdr *section,
                                                   const Elf_Sym *symb,
                                                   bool &IsDefault) const {
  StringRef StrTab;
  if (section) {
    ErrorOr<StringRef> StrTabOrErr = getStringTable(section);
    if (std::error_code EC = StrTabOrErr.getError())
      return EC;
    StrTab = *StrTabOrErr;
  }
  // Handle non-dynamic symbols.
  if (section != DotDynSymSec && section != nullptr) {
    // Non-dynamic symbols can have versions in their names
    // A name of the form 'foo@V1' indicates version 'V1', non-default.
    // A name of the form 'foo@@V2' indicates version 'V2', default version.
    ErrorOr<StringRef> SymName = symb->getName(StrTab);
    if (!SymName)
      return SymName;
    StringRef Name = *SymName;
    size_t atpos = Name.find('@');
    if (atpos == StringRef::npos) {
      IsDefault = false;
      return StringRef("");
    }
    ++atpos;
    if (atpos < Name.size() && Name[atpos] == '@') {
      IsDefault = true;
      ++atpos;
    } else {
      IsDefault = false;
    }
    return Name.substr(atpos);
  }

  // This is a dynamic symbol. Look in the GNU symbol version table.
  if (!dot_gnu_version_sec) {
    // No version table.
    IsDefault = false;
    return StringRef("");
  }

  // Determine the position in the symbol table of this entry.
  size_t entry_index =
      (reinterpret_cast<uintptr_t>(symb) - DotDynSymSec->sh_offset -
       reinterpret_cast<uintptr_t>(base())) /
      sizeof(Elf_Sym);

  // Get the corresponding version index entry
  const Elf_Versym *vs = getEntry<Elf_Versym>(dot_gnu_version_sec, entry_index);
  size_t version_index = vs->vs_index & ELF::VERSYM_VERSION;

  // Special markers for unversioned symbols.
  if (version_index == ELF::VER_NDX_LOCAL ||
      version_index == ELF::VER_NDX_GLOBAL) {
    IsDefault = false;
    return StringRef("");
  }

  // Lookup this symbol in the version table
  LoadVersionMap();
  if (version_index >= VersionMap.size() || VersionMap[version_index].isNull())
    return object_error::parse_failed;
  const VersionMapEntry &entry = VersionMap[version_index];

  // Get the version name string
  size_t name_offset;
  if (entry.isVerdef()) {
    // The first Verdaux entry holds the name.
    name_offset = entry.getVerdef()->getAux()->vda_name;
  } else {
    name_offset = entry.getVernaux()->vna_name;
  }

  // Set IsDefault
  if (entry.isVerdef()) {
    IsDefault = !(vs->vs_index & ELF::VERSYM_HIDDEN);
  } else {
    IsDefault = false;
  }

  if (name_offset >= DynStrRegion.Size)
    return object_error::parse_failed;
  return StringRef(getDynamicString(name_offset));
}

/// This function returns the hash value for a symbol in the .dynsym section
/// Name of the API remains consistent as specified in the libelf
/// REF : http://www.sco.com/developers/gabi/latest/ch5.dynamic.html#hash
static inline unsigned elf_hash(StringRef &symbolName) {
  unsigned h = 0, g;
  for (unsigned i = 0, j = symbolName.size(); i < j; i++) {
    h = (h << 4) + symbolName[i];
    g = h & 0xf0000000L;
    if (g != 0)
      h ^= g >> 24;
    h &= ~g;
  }
  return h;
}
} // end namespace object
} // end namespace llvm

#endif
