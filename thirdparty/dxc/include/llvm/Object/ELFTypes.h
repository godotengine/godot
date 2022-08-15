//===- ELFTypes.h - Endian specific types for ELF ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ELFTYPES_H
#define LLVM_OBJECT_ELFTYPES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {
namespace object {

using support::endianness;

template <endianness target_endianness, bool is64Bits> struct ELFType {
  static const endianness TargetEndianness = target_endianness;
  static const bool Is64Bits = is64Bits;
};

typedef ELFType<support::little, false> ELF32LE;
typedef ELFType<support::big, false> ELF32BE;
typedef ELFType<support::little, true> ELF64LE;
typedef ELFType<support::big, true> ELF64BE;

// Use an alignment of 2 for the typedefs since that is the worst case for
// ELF files in archives.

// Templates to choose Elf_Addr and Elf_Off depending on is64Bits.
template <endianness target_endianness> struct ELFDataTypeTypedefHelperCommon {
  typedef support::detail::packed_endian_specific_integral<
      uint16_t, target_endianness, 2> Elf_Half;
  typedef support::detail::packed_endian_specific_integral<
      uint32_t, target_endianness, 2> Elf_Word;
  typedef support::detail::packed_endian_specific_integral<
      int32_t, target_endianness, 2> Elf_Sword;
  typedef support::detail::packed_endian_specific_integral<
      uint64_t, target_endianness, 2> Elf_Xword;
  typedef support::detail::packed_endian_specific_integral<
      int64_t, target_endianness, 2> Elf_Sxword;
};

template <class ELFT> struct ELFDataTypeTypedefHelper;

/// ELF 32bit types.
template <endianness TargetEndianness>
struct ELFDataTypeTypedefHelper<ELFType<TargetEndianness, false>>
    : ELFDataTypeTypedefHelperCommon<TargetEndianness> {
  typedef uint32_t value_type;
  typedef support::detail::packed_endian_specific_integral<
      value_type, TargetEndianness, 2> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral<
      value_type, TargetEndianness, 2> Elf_Off;
};

/// ELF 64bit types.
template <endianness TargetEndianness>
struct ELFDataTypeTypedefHelper<ELFType<TargetEndianness, true>>
    : ELFDataTypeTypedefHelperCommon<TargetEndianness> {
  typedef uint64_t value_type;
  typedef support::detail::packed_endian_specific_integral<
      value_type, TargetEndianness, 2> Elf_Addr;
  typedef support::detail::packed_endian_specific_integral<
      value_type, TargetEndianness, 2> Elf_Off;
};

// I really don't like doing this, but the alternative is copypasta.
#define LLVM_ELF_IMPORT_TYPES(E, W)                                            \
  typedef typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Addr Elf_Addr; \
  typedef typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Off Elf_Off;   \
  typedef typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Half Elf_Half; \
  typedef typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Word Elf_Word; \
  typedef                                                                      \
      typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Sword Elf_Sword;   \
  typedef                                                                      \
      typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Xword Elf_Xword;   \
  typedef                                                                      \
      typename ELFDataTypeTypedefHelper<ELFType<E, W>>::Elf_Sxword Elf_Sxword;

#define LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)                                       \
  LLVM_ELF_IMPORT_TYPES(ELFT::TargetEndianness, ELFT::Is64Bits)

// Section header.
template <class ELFT> struct Elf_Shdr_Base;

template <endianness TargetEndianness>
struct Elf_Shdr_Base<ELFType<TargetEndianness, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Word sh_name;      // Section name (index into string table)
  Elf_Word sh_type;      // Section type (SHT_*)
  Elf_Word sh_flags;     // Section flags (SHF_*)
  Elf_Addr sh_addr;      // Address where section is to be loaded
  Elf_Off sh_offset;     // File offset of section data, in bytes
  Elf_Word sh_size;      // Size of section, in bytes
  Elf_Word sh_link;      // Section type-specific header table index link
  Elf_Word sh_info;      // Section type-specific extra information
  Elf_Word sh_addralign; // Section address alignment
  Elf_Word sh_entsize;   // Size of records contained within the section
};

template <endianness TargetEndianness>
struct Elf_Shdr_Base<ELFType<TargetEndianness, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Word sh_name;       // Section name (index into string table)
  Elf_Word sh_type;       // Section type (SHT_*)
  Elf_Xword sh_flags;     // Section flags (SHF_*)
  Elf_Addr sh_addr;       // Address where section is to be loaded
  Elf_Off sh_offset;      // File offset of section data, in bytes
  Elf_Xword sh_size;      // Size of section, in bytes
  Elf_Word sh_link;       // Section type-specific header table index link
  Elf_Word sh_info;       // Section type-specific extra information
  Elf_Xword sh_addralign; // Section address alignment
  Elf_Xword sh_entsize;   // Size of records contained within the section
};

template <class ELFT>
struct Elf_Shdr_Impl : Elf_Shdr_Base<ELFT> {
  using Elf_Shdr_Base<ELFT>::sh_entsize;
  using Elf_Shdr_Base<ELFT>::sh_size;

  /// @brief Get the number of entities this section contains if it has any.
  unsigned getEntityCount() const {
    if (sh_entsize == 0)
      return 0;
    return sh_size / sh_entsize;
  }
};

template <class ELFT> struct Elf_Sym_Base;

template <endianness TargetEndianness>
struct Elf_Sym_Base<ELFType<TargetEndianness, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Word st_name;       // Symbol name (index into string table)
  Elf_Addr st_value;      // Value or address associated with the symbol
  Elf_Word st_size;       // Size of the symbol
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half st_shndx;      // Which section (header table index) it's defined in
};

template <endianness TargetEndianness>
struct Elf_Sym_Base<ELFType<TargetEndianness, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Word st_name;       // Symbol name (index into string table)
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf_Half st_shndx;      // Which section (header table index) it's defined in
  Elf_Addr st_value;      // Value or address associated with the symbol
  Elf_Xword st_size;      // Size of the symbol
};

template <class ELFT>
struct Elf_Sym_Impl : Elf_Sym_Base<ELFT> {
  using Elf_Sym_Base<ELFT>::st_info;
  using Elf_Sym_Base<ELFT>::st_shndx;
  using Elf_Sym_Base<ELFT>::st_other;
  using Elf_Sym_Base<ELFT>::st_value;

  // These accessors and mutators correspond to the ELF32_ST_BIND,
  // ELF32_ST_TYPE, and ELF32_ST_INFO macros defined in the ELF specification:
  unsigned char getBinding() const { return st_info >> 4; }
  unsigned char getType() const { return st_info & 0x0f; }
  uint64_t getValue() const { return st_value; }
  void setBinding(unsigned char b) { setBindingAndType(b, getType()); }
  void setType(unsigned char t) { setBindingAndType(getBinding(), t); }
  void setBindingAndType(unsigned char b, unsigned char t) {
    st_info = (b << 4) + (t & 0x0f);
  }

  /// Access to the STV_xxx flag stored in the first two bits of st_other.
  /// STV_DEFAULT: 0
  /// STV_INTERNAL: 1
  /// STV_HIDDEN: 2
  /// STV_PROTECTED: 3
  unsigned char getVisibility() const { return st_other & 0x3; }
  void setVisibility(unsigned char v) {
    assert(v < 4 && "Invalid value for visibility");
    st_other = (st_other & ~0x3) | v;
  }

  bool isAbsolute() const { return st_shndx == ELF::SHN_ABS; }
  bool isCommon() const {
    return getType() == ELF::STT_COMMON || st_shndx == ELF::SHN_COMMON;
  }
  bool isDefined() const { return !isUndefined(); }
  bool isProcessorSpecific() const {
    return st_shndx >= ELF::SHN_LOPROC && st_shndx <= ELF::SHN_HIPROC;
  }
  bool isOSSpecific() const {
    return st_shndx >= ELF::SHN_LOOS && st_shndx <= ELF::SHN_HIOS;
  }
  bool isReserved() const {
    // ELF::SHN_HIRESERVE is 0xffff so st_shndx <= ELF::SHN_HIRESERVE is always
    // true and some compilers warn about it.
    return st_shndx >= ELF::SHN_LORESERVE;
  }
  bool isUndefined() const { return st_shndx == ELF::SHN_UNDEF; }
  bool isExternal() const {
    return getBinding() != ELF::STB_LOCAL;
  }

  ErrorOr<StringRef> getName(StringRef StrTab) const;
};

template <class ELFT>
ErrorOr<StringRef> Elf_Sym_Impl<ELFT>::getName(StringRef StrTab) const {
  uint32_t Offset = this->st_name;
  if (Offset >= StrTab.size())
    return object_error::parse_failed;
  return StringRef(StrTab.data() + Offset);
}

/// Elf_Versym: This is the structure of entries in the SHT_GNU_versym section
/// (.gnu.version). This structure is identical for ELF32 and ELF64.
template <class ELFT>
struct Elf_Versym_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Half vs_index; // Version index with flags (e.g. VERSYM_HIDDEN)
};

template <class ELFT> struct Elf_Verdaux_Impl;

/// Elf_Verdef: This is the structure of entries in the SHT_GNU_verdef section
/// (.gnu.version_d). This structure is identical for ELF32 and ELF64.
template <class ELFT>
struct Elf_Verdef_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  typedef Elf_Verdaux_Impl<ELFT> Elf_Verdaux;
  Elf_Half vd_version; // Version of this structure (e.g. VER_DEF_CURRENT)
  Elf_Half vd_flags;   // Bitwise flags (VER_DEF_*)
  Elf_Half vd_ndx;     // Version index, used in .gnu.version entries
  Elf_Half vd_cnt;     // Number of Verdaux entries
  Elf_Word vd_hash;    // Hash of name
  Elf_Word vd_aux;     // Offset to the first Verdaux entry (in bytes)
  Elf_Word vd_next;    // Offset to the next Verdef entry (in bytes)

  /// Get the first Verdaux entry for this Verdef.
  const Elf_Verdaux *getAux() const {
    return reinterpret_cast<const Elf_Verdaux *>((const char *)this + vd_aux);
  }
};

/// Elf_Verdaux: This is the structure of auxiliary data in the SHT_GNU_verdef
/// section (.gnu.version_d). This structure is identical for ELF32 and ELF64.
template <class ELFT>
struct Elf_Verdaux_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Word vda_name; // Version name (offset in string table)
  Elf_Word vda_next; // Offset to next Verdaux entry (in bytes)
};

/// Elf_Verneed: This is the structure of entries in the SHT_GNU_verneed
/// section (.gnu.version_r). This structure is identical for ELF32 and ELF64.
template <class ELFT>
struct Elf_Verneed_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Half vn_version; // Version of this structure (e.g. VER_NEED_CURRENT)
  Elf_Half vn_cnt;     // Number of associated Vernaux entries
  Elf_Word vn_file;    // Library name (string table offset)
  Elf_Word vn_aux;     // Offset to first Vernaux entry (in bytes)
  Elf_Word vn_next;    // Offset to next Verneed entry (in bytes)
};

/// Elf_Vernaux: This is the structure of auxiliary data in SHT_GNU_verneed
/// section (.gnu.version_r). This structure is identical for ELF32 and ELF64.
template <class ELFT>
struct Elf_Vernaux_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Word vna_hash;  // Hash of dependency name
  Elf_Half vna_flags; // Bitwise Flags (VER_FLAG_*)
  Elf_Half vna_other; // Version index, used in .gnu.version entries
  Elf_Word vna_name;  // Dependency name
  Elf_Word vna_next;  // Offset to next Vernaux entry (in bytes)
};

/// Elf_Dyn_Base: This structure matches the form of entries in the dynamic
///               table section (.dynamic) look like.
template <class ELFT> struct Elf_Dyn_Base;

template <endianness TargetEndianness>
struct Elf_Dyn_Base<ELFType<TargetEndianness, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Sword d_tag;
  union {
    Elf_Word d_val;
    Elf_Addr d_ptr;
  } d_un;
};

template <endianness TargetEndianness>
struct Elf_Dyn_Base<ELFType<TargetEndianness, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Sxword d_tag;
  union {
    Elf_Xword d_val;
    Elf_Addr d_ptr;
  } d_un;
};

/// Elf_Dyn_Impl: This inherits from Elf_Dyn_Base, adding getters and setters.
template <class ELFT>
struct Elf_Dyn_Impl : Elf_Dyn_Base<ELFT> {
  using Elf_Dyn_Base<ELFT>::d_tag;
  using Elf_Dyn_Base<ELFT>::d_un;
  int64_t getTag() const { return d_tag; }
  uint64_t getVal() const { return d_un.d_val; }
  uint64_t getPtr() const { return d_un.d_ptr; }
};

// Elf_Rel: Elf Relocation
template <class ELFT, bool isRela> struct Elf_Rel_Impl;

template <endianness TargetEndianness>
struct Elf_Rel_Impl<ELFType<TargetEndianness, false>, false> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Addr r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Word r_info;   // Symbol table index and type of relocation to apply

  uint32_t getRInfo(bool isMips64EL) const {
    assert(!isMips64EL);
    return r_info;
  }
  void setRInfo(uint32_t R, bool IsMips64EL) {
    assert(!IsMips64EL);
    r_info = R;
  }

  // These accessors and mutators correspond to the ELF32_R_SYM, ELF32_R_TYPE,
  // and ELF32_R_INFO macros defined in the ELF specification:
  uint32_t getSymbol(bool isMips64EL) const {
    return this->getRInfo(isMips64EL) >> 8;
  }
  unsigned char getType(bool isMips64EL) const {
    return (unsigned char)(this->getRInfo(isMips64EL) & 0x0ff);
  }
  void setSymbol(uint32_t s, bool IsMips64EL) {
    setSymbolAndType(s, getType(), IsMips64EL);
  }
  void setType(unsigned char t, bool IsMips64EL) {
    setSymbolAndType(getSymbol(), t, IsMips64EL);
  }
  void setSymbolAndType(uint32_t s, unsigned char t, bool IsMips64EL) {
    this->setRInfo((s << 8) + t, IsMips64EL);
  }
};

template <endianness TargetEndianness>
struct Elf_Rel_Impl<ELFType<TargetEndianness, false>, true>
    : public Elf_Rel_Impl<ELFType<TargetEndianness, false>, false> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Sword r_addend; // Compute value for relocatable field by adding this
};

template <endianness TargetEndianness>
struct Elf_Rel_Impl<ELFType<TargetEndianness, true>, false> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Addr r_offset; // Location (file byte offset, or program virtual addr)
  Elf_Xword r_info;  // Symbol table index and type of relocation to apply

  uint64_t getRInfo(bool isMips64EL) const {
    uint64_t t = r_info;
    if (!isMips64EL)
      return t;
    // Mips64 little endian has a "special" encoding of r_info. Instead of one
    // 64 bit little endian number, it is a little endian 32 bit number followed
    // by a 32 bit big endian number.
    return (t << 32) | ((t >> 8) & 0xff000000) | ((t >> 24) & 0x00ff0000) |
           ((t >> 40) & 0x0000ff00) | ((t >> 56) & 0x000000ff);
  }
  void setRInfo(uint64_t R, bool IsMips64EL) {
    if (IsMips64EL)
      r_info = (R >> 32) | ((R & 0xff000000) << 8) | ((R & 0x00ff0000) << 24) |
               ((R & 0x0000ff00) << 40) | ((R & 0x000000ff) << 56);
    else
      r_info = R;
  }

  // These accessors and mutators correspond to the ELF64_R_SYM, ELF64_R_TYPE,
  // and ELF64_R_INFO macros defined in the ELF specification:
  uint32_t getSymbol(bool isMips64EL) const {
    return (uint32_t)(this->getRInfo(isMips64EL) >> 32);
  }
  uint32_t getType(bool isMips64EL) const {
    return (uint32_t)(this->getRInfo(isMips64EL) & 0xffffffffL);
  }
  void setSymbol(uint32_t s, bool IsMips64EL) {
    setSymbolAndType(s, getType(), IsMips64EL);
  }
  void setType(uint32_t t, bool IsMips64EL) {
    setSymbolAndType(getSymbol(), t, IsMips64EL);
  }
  void setSymbolAndType(uint32_t s, uint32_t t, bool IsMips64EL) {
    this->setRInfo(((uint64_t)s << 32) + (t & 0xffffffffL), IsMips64EL);
  }
};

template <endianness TargetEndianness>
struct Elf_Rel_Impl<ELFType<TargetEndianness, true>, true>
    : public Elf_Rel_Impl<ELFType<TargetEndianness, true>, false> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Sxword r_addend; // Compute value for relocatable field by adding this.
};

template <class ELFT>
struct Elf_Ehdr_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  unsigned char e_ident[ELF::EI_NIDENT]; // ELF Identification bytes
  Elf_Half e_type;                       // Type of file (see ET_*)
  Elf_Half e_machine;   // Required architecture for this file (see EM_*)
  Elf_Word e_version;   // Must be equal to 1
  Elf_Addr e_entry;     // Address to jump to in order to start program
  Elf_Off e_phoff;      // Program header table's file offset, in bytes
  Elf_Off e_shoff;      // Section header table's file offset, in bytes
  Elf_Word e_flags;     // Processor-specific flags
  Elf_Half e_ehsize;    // Size of ELF header, in bytes
  Elf_Half e_phentsize; // Size of an entry in the program header table
  Elf_Half e_phnum;     // Number of entries in the program header table
  Elf_Half e_shentsize; // Size of an entry in the section header table
  Elf_Half e_shnum;     // Number of entries in the section header table
  Elf_Half e_shstrndx;  // Section header table index of section name
                        // string table
  bool checkMagic() const {
    return (memcmp(e_ident, ELF::ElfMagic, strlen(ELF::ElfMagic))) == 0;
  }
  unsigned char getFileClass() const { return e_ident[ELF::EI_CLASS]; }
  unsigned char getDataEncoding() const { return e_ident[ELF::EI_DATA]; }
};

template <class ELFT> struct Elf_Phdr_Impl;

template <endianness TargetEndianness>
struct Elf_Phdr_Impl<ELFType<TargetEndianness, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Word p_type;   // Type of segment
  Elf_Off p_offset;  // FileOffset where segment is located, in bytes
  Elf_Addr p_vaddr;  // Virtual Address of beginning of segment
  Elf_Addr p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf_Word p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf_Word p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf_Word p_flags;  // Segment flags
  Elf_Word p_align;  // Segment alignment constraint
};

template <endianness TargetEndianness>
struct Elf_Phdr_Impl<ELFType<TargetEndianness, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Word p_type;    // Type of segment
  Elf_Word p_flags;   // Segment flags
  Elf_Off p_offset;   // FileOffset where segment is located, in bytes
  Elf_Addr p_vaddr;   // Virtual Address of beginning of segment
  Elf_Addr p_paddr;   // Physical address of beginning of segment (OS-specific)
  Elf_Xword p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf_Xword p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf_Xword p_align;  // Segment alignment constraint
};

// ELFT needed for endianess.
template <class ELFT>
struct Elf_Hash_Impl {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Word nbucket;
  Elf_Word nchain;

  ArrayRef<Elf_Word> buckets() const {
    return ArrayRef<Elf_Word>(&nbucket + 2, &nbucket + 2 + nbucket);
  }

  ArrayRef<Elf_Word> chains() const {
    return ArrayRef<Elf_Word>(&nbucket + 2 + nbucket,
                              &nbucket + 2 + nbucket + nchain);
  }
};

// MIPS .reginfo section
template <class ELFT>
struct Elf_Mips_RegInfo;

template <llvm::support::endianness TargetEndianness>
struct Elf_Mips_RegInfo<ELFType<TargetEndianness, false>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, false)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Addr ri_gp_value;    // gp register value
};

template <llvm::support::endianness TargetEndianness>
struct Elf_Mips_RegInfo<ELFType<TargetEndianness, true>> {
  LLVM_ELF_IMPORT_TYPES(TargetEndianness, true)
  Elf_Word ri_gprmask;     // bit-mask of used general registers
  Elf_Word ri_pad;         // unused padding field
  Elf_Word ri_cprmask[4];  // bit-mask of used co-processor registers
  Elf_Addr ri_gp_value;    // gp register value
};

// .MIPS.options section
template <class ELFT> struct Elf_Mips_Options {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  uint8_t kind;     // Determines interpretation of variable part of descriptor
  uint8_t size;     // Byte size of descriptor, including this header
  Elf_Half section; // Section header index of section affected,
                    // or 0 for global options
  Elf_Word info;    // Kind-specific information

  const Elf_Mips_RegInfo<ELFT> &getRegInfo() const {
    assert(kind == llvm::ELF::ODK_REGINFO);
    return *reinterpret_cast<const Elf_Mips_RegInfo<ELFT> *>(
               (const uint8_t *)this + sizeof(Elf_Mips_Options));
  }
};

// .MIPS.abiflags section content
template <class ELFT> struct Elf_Mips_ABIFlags {
  LLVM_ELF_IMPORT_TYPES_ELFT(ELFT)
  Elf_Half version;  // Version of the structure
  uint8_t isa_level; // ISA level: 1-5, 32, and 64
  uint8_t isa_rev;   // ISA revision (0 for MIPS I - MIPS V)
  uint8_t gpr_size;  // General purpose registers size
  uint8_t cpr1_size; // Co-processor 1 registers size
  uint8_t cpr2_size; // Co-processor 2 registers size
  uint8_t fp_abi;    // Floating-point ABI flag
  Elf_Word isa_ext;  // Processor-specific extension
  Elf_Word ases;     // ASEs flags
  Elf_Word flags1;   // General flags
  Elf_Word flags2;   // General flags
};

} // end namespace object.
} // end namespace llvm.

#endif
