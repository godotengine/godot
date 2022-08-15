//===- COFF.h - COFF object file implementation -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the COFFObjectFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_COFF_H
#define LLVM_OBJECT_COFF_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorOr.h"

namespace llvm {
template <typename T> class ArrayRef;

namespace object {
class ImportDirectoryEntryRef;
class DelayImportDirectoryEntryRef;
class ExportDirectoryEntryRef;
class ImportedSymbolRef;
class BaseRelocRef;
typedef content_iterator<ImportDirectoryEntryRef> import_directory_iterator;
typedef content_iterator<DelayImportDirectoryEntryRef>
    delay_import_directory_iterator;
typedef content_iterator<ExportDirectoryEntryRef> export_directory_iterator;
typedef content_iterator<ImportedSymbolRef> imported_symbol_iterator;
typedef content_iterator<BaseRelocRef> base_reloc_iterator;

/// The DOS compatible header at the front of all PE/COFF executables.
struct dos_header {
  char                 Magic[2];
  support::ulittle16_t UsedBytesInTheLastPage;
  support::ulittle16_t FileSizeInPages;
  support::ulittle16_t NumberOfRelocationItems;
  support::ulittle16_t HeaderSizeInParagraphs;
  support::ulittle16_t MinimumExtraParagraphs;
  support::ulittle16_t MaximumExtraParagraphs;
  support::ulittle16_t InitialRelativeSS;
  support::ulittle16_t InitialSP;
  support::ulittle16_t Checksum;
  support::ulittle16_t InitialIP;
  support::ulittle16_t InitialRelativeCS;
  support::ulittle16_t AddressOfRelocationTable;
  support::ulittle16_t OverlayNumber;
  support::ulittle16_t Reserved[4];
  support::ulittle16_t OEMid;
  support::ulittle16_t OEMinfo;
  support::ulittle16_t Reserved2[10];
  support::ulittle32_t AddressOfNewExeHeader;
};

struct coff_file_header {
  support::ulittle16_t Machine;
  support::ulittle16_t NumberOfSections;
  support::ulittle32_t TimeDateStamp;
  support::ulittle32_t PointerToSymbolTable;
  support::ulittle32_t NumberOfSymbols;
  support::ulittle16_t SizeOfOptionalHeader;
  support::ulittle16_t Characteristics;

  bool isImportLibrary() const { return NumberOfSections == 0xffff; }
};

struct coff_bigobj_file_header {
  support::ulittle16_t Sig1;
  support::ulittle16_t Sig2;
  support::ulittle16_t Version;
  support::ulittle16_t Machine;
  support::ulittle32_t TimeDateStamp;
  uint8_t              UUID[16];
  support::ulittle32_t unused1;
  support::ulittle32_t unused2;
  support::ulittle32_t unused3;
  support::ulittle32_t unused4;
  support::ulittle32_t NumberOfSections;
  support::ulittle32_t PointerToSymbolTable;
  support::ulittle32_t NumberOfSymbols;
};

/// The 32-bit PE header that follows the COFF header.
struct pe32_header {
  support::ulittle16_t Magic;
  uint8_t MajorLinkerVersion;
  uint8_t MinorLinkerVersion;
  support::ulittle32_t SizeOfCode;
  support::ulittle32_t SizeOfInitializedData;
  support::ulittle32_t SizeOfUninitializedData;
  support::ulittle32_t AddressOfEntryPoint;
  support::ulittle32_t BaseOfCode;
  support::ulittle32_t BaseOfData;
  support::ulittle32_t ImageBase;
  support::ulittle32_t SectionAlignment;
  support::ulittle32_t FileAlignment;
  support::ulittle16_t MajorOperatingSystemVersion;
  support::ulittle16_t MinorOperatingSystemVersion;
  support::ulittle16_t MajorImageVersion;
  support::ulittle16_t MinorImageVersion;
  support::ulittle16_t MajorSubsystemVersion;
  support::ulittle16_t MinorSubsystemVersion;
  support::ulittle32_t Win32VersionValue;
  support::ulittle32_t SizeOfImage;
  support::ulittle32_t SizeOfHeaders;
  support::ulittle32_t CheckSum;
  support::ulittle16_t Subsystem;
  // FIXME: This should be DllCharacteristics.
  support::ulittle16_t DLLCharacteristics;
  support::ulittle32_t SizeOfStackReserve;
  support::ulittle32_t SizeOfStackCommit;
  support::ulittle32_t SizeOfHeapReserve;
  support::ulittle32_t SizeOfHeapCommit;
  support::ulittle32_t LoaderFlags;
  // FIXME: This should be NumberOfRvaAndSizes.
  support::ulittle32_t NumberOfRvaAndSize;
};

/// The 64-bit PE header that follows the COFF header.
struct pe32plus_header {
  support::ulittle16_t Magic;
  uint8_t MajorLinkerVersion;
  uint8_t MinorLinkerVersion;
  support::ulittle32_t SizeOfCode;
  support::ulittle32_t SizeOfInitializedData;
  support::ulittle32_t SizeOfUninitializedData;
  support::ulittle32_t AddressOfEntryPoint;
  support::ulittle32_t BaseOfCode;
  support::ulittle64_t ImageBase;
  support::ulittle32_t SectionAlignment;
  support::ulittle32_t FileAlignment;
  support::ulittle16_t MajorOperatingSystemVersion;
  support::ulittle16_t MinorOperatingSystemVersion;
  support::ulittle16_t MajorImageVersion;
  support::ulittle16_t MinorImageVersion;
  support::ulittle16_t MajorSubsystemVersion;
  support::ulittle16_t MinorSubsystemVersion;
  support::ulittle32_t Win32VersionValue;
  support::ulittle32_t SizeOfImage;
  support::ulittle32_t SizeOfHeaders;
  support::ulittle32_t CheckSum;
  support::ulittle16_t Subsystem;
  support::ulittle16_t DLLCharacteristics;
  support::ulittle64_t SizeOfStackReserve;
  support::ulittle64_t SizeOfStackCommit;
  support::ulittle64_t SizeOfHeapReserve;
  support::ulittle64_t SizeOfHeapCommit;
  support::ulittle32_t LoaderFlags;
  support::ulittle32_t NumberOfRvaAndSize;
};

struct data_directory {
  support::ulittle32_t RelativeVirtualAddress;
  support::ulittle32_t Size;
};

struct import_directory_table_entry {
  support::ulittle32_t ImportLookupTableRVA;
  support::ulittle32_t TimeDateStamp;
  support::ulittle32_t ForwarderChain;
  support::ulittle32_t NameRVA;
  support::ulittle32_t ImportAddressTableRVA;
};

template <typename IntTy>
struct import_lookup_table_entry {
  IntTy Data;

  bool isOrdinal() const { return Data < 0; }

  uint16_t getOrdinal() const {
    assert(isOrdinal() && "ILT entry is not an ordinal!");
    return Data & 0xFFFF;
  }

  uint32_t getHintNameRVA() const {
    assert(!isOrdinal() && "ILT entry is not a Hint/Name RVA!");
    return Data & 0xFFFFFFFF;
  }
};

typedef import_lookup_table_entry<support::little32_t>
    import_lookup_table_entry32;
typedef import_lookup_table_entry<support::little64_t>
    import_lookup_table_entry64;

struct delay_import_directory_table_entry {
  // dumpbin reports this field as "Characteristics" instead of "Attributes".
  support::ulittle32_t Attributes;
  support::ulittle32_t Name;
  support::ulittle32_t ModuleHandle;
  support::ulittle32_t DelayImportAddressTable;
  support::ulittle32_t DelayImportNameTable;
  support::ulittle32_t BoundDelayImportTable;
  support::ulittle32_t UnloadDelayImportTable;
  support::ulittle32_t TimeStamp;
};

struct export_directory_table_entry {
  support::ulittle32_t ExportFlags;
  support::ulittle32_t TimeDateStamp;
  support::ulittle16_t MajorVersion;
  support::ulittle16_t MinorVersion;
  support::ulittle32_t NameRVA;
  support::ulittle32_t OrdinalBase;
  support::ulittle32_t AddressTableEntries;
  support::ulittle32_t NumberOfNamePointers;
  support::ulittle32_t ExportAddressTableRVA;
  support::ulittle32_t NamePointerRVA;
  support::ulittle32_t OrdinalTableRVA;
};

union export_address_table_entry {
  support::ulittle32_t ExportRVA;
  support::ulittle32_t ForwarderRVA;
};

typedef support::ulittle32_t export_name_pointer_table_entry;
typedef support::ulittle16_t export_ordinal_table_entry;

struct StringTableOffset {
  support::ulittle32_t Zeroes;
  support::ulittle32_t Offset;
};

template <typename SectionNumberType>
struct coff_symbol {
  union {
    char ShortName[COFF::NameSize];
    StringTableOffset Offset;
  } Name;

  support::ulittle32_t Value;
  SectionNumberType SectionNumber;

  support::ulittle16_t Type;

  uint8_t StorageClass;
  uint8_t NumberOfAuxSymbols;
};

typedef coff_symbol<support::ulittle16_t> coff_symbol16;
typedef coff_symbol<support::ulittle32_t> coff_symbol32;

// Contains only common parts of coff_symbol16 and coff_symbol32.
struct coff_symbol_generic {
  union {
    char ShortName[COFF::NameSize];
    StringTableOffset Offset;
  } Name;
  support::ulittle32_t Value;
};

class COFFSymbolRef {
public:
  COFFSymbolRef(const coff_symbol16 *CS) : CS16(CS), CS32(nullptr) {}
  COFFSymbolRef(const coff_symbol32 *CS) : CS16(nullptr), CS32(CS) {}
  COFFSymbolRef() : CS16(nullptr), CS32(nullptr) {}

  const void *getRawPtr() const {
    return CS16 ? static_cast<const void *>(CS16) : CS32;
  }

  const coff_symbol_generic *getGeneric() const {
    if (CS16)
      return reinterpret_cast<const coff_symbol_generic *>(CS16);
    return reinterpret_cast<const coff_symbol_generic *>(CS32);
  }

  friend bool operator<(COFFSymbolRef A, COFFSymbolRef B) {
    return A.getRawPtr() < B.getRawPtr();
  }

  bool isBigObj() const {
    if (CS16)
      return false;
    if (CS32)
      return true;
    llvm_unreachable("COFFSymbolRef points to nothing!");
  }

  const char *getShortName() const {
    return CS16 ? CS16->Name.ShortName : CS32->Name.ShortName;
  }

  const StringTableOffset &getStringTableOffset() const {
    assert(isSet() && "COFFSymbolRef points to nothing!");
    return CS16 ? CS16->Name.Offset : CS32->Name.Offset;
  }

  uint32_t getValue() const { return CS16 ? CS16->Value : CS32->Value; }

  int32_t getSectionNumber() const {
    assert(isSet() && "COFFSymbolRef points to nothing!");
    if (CS16) {
      // Reserved sections are returned as negative numbers.
      if (CS16->SectionNumber <= COFF::MaxNumberOfSections16)
        return CS16->SectionNumber;
      return static_cast<int16_t>(CS16->SectionNumber);
    }
    return static_cast<int32_t>(CS32->SectionNumber);
  }

  uint16_t getType() const {
    assert(isSet() && "COFFSymbolRef points to nothing!");
    return CS16 ? CS16->Type : CS32->Type;
  }

  uint8_t getStorageClass() const {
    assert(isSet() && "COFFSymbolRef points to nothing!");
    return CS16 ? CS16->StorageClass : CS32->StorageClass;
  }

  uint8_t getNumberOfAuxSymbols() const {
    assert(isSet() && "COFFSymbolRef points to nothing!");
    return CS16 ? CS16->NumberOfAuxSymbols : CS32->NumberOfAuxSymbols;
  }

  uint8_t getBaseType() const { return getType() & 0x0F; }

  uint8_t getComplexType() const {
    return (getType() & 0xF0) >> COFF::SCT_COMPLEX_TYPE_SHIFT;
  }

  bool isAbsolute() const {
    return getSectionNumber() == -1;
  }

  bool isExternal() const {
    return getStorageClass() == COFF::IMAGE_SYM_CLASS_EXTERNAL;
  }

  bool isCommon() const {
    return isExternal() && getSectionNumber() == COFF::IMAGE_SYM_UNDEFINED &&
           getValue() != 0;
  }

  bool isUndefined() const {
    return isExternal() && getSectionNumber() == COFF::IMAGE_SYM_UNDEFINED &&
           getValue() == 0;
  }

  bool isWeakExternal() const {
    return getStorageClass() == COFF::IMAGE_SYM_CLASS_WEAK_EXTERNAL;
  }

  bool isFunctionDefinition() const {
    return isExternal() && getBaseType() == COFF::IMAGE_SYM_TYPE_NULL &&
           getComplexType() == COFF::IMAGE_SYM_DTYPE_FUNCTION &&
           !COFF::isReservedSectionNumber(getSectionNumber());
  }

  bool isFunctionLineInfo() const {
    return getStorageClass() == COFF::IMAGE_SYM_CLASS_FUNCTION;
  }

  bool isAnyUndefined() const {
    return isUndefined() || isWeakExternal();
  }

  bool isFileRecord() const {
    return getStorageClass() == COFF::IMAGE_SYM_CLASS_FILE;
  }

  bool isSection() const {
    return getStorageClass() == COFF::IMAGE_SYM_CLASS_SECTION;
  }

  bool isSectionDefinition() const {
    // C++/CLI creates external ABS symbols for non-const appdomain globals.
    // These are also followed by an auxiliary section definition.
    bool isAppdomainGlobal =
        getStorageClass() == COFF::IMAGE_SYM_CLASS_EXTERNAL &&
        getSectionNumber() == COFF::IMAGE_SYM_ABSOLUTE;
    bool isOrdinarySection = getStorageClass() == COFF::IMAGE_SYM_CLASS_STATIC;
    if (!getNumberOfAuxSymbols())
      return false;
    return isAppdomainGlobal || isOrdinarySection;
  }

  bool isCLRToken() const {
    return getStorageClass() == COFF::IMAGE_SYM_CLASS_CLR_TOKEN;
  }

private:
  bool isSet() const { return CS16 || CS32; }

  const coff_symbol16 *CS16;
  const coff_symbol32 *CS32;
};

struct coff_section {
  char Name[COFF::NameSize];
  support::ulittle32_t VirtualSize;
  support::ulittle32_t VirtualAddress;
  support::ulittle32_t SizeOfRawData;
  support::ulittle32_t PointerToRawData;
  support::ulittle32_t PointerToRelocations;
  support::ulittle32_t PointerToLinenumbers;
  support::ulittle16_t NumberOfRelocations;
  support::ulittle16_t NumberOfLinenumbers;
  support::ulittle32_t Characteristics;

  // Returns true if the actual number of relocations is stored in
  // VirtualAddress field of the first relocation table entry.
  bool hasExtendedRelocations() const {
    return (Characteristics & COFF::IMAGE_SCN_LNK_NRELOC_OVFL) &&
           NumberOfRelocations == UINT16_MAX;
  }
};

struct coff_relocation {
  support::ulittle32_t VirtualAddress;
  support::ulittle32_t SymbolTableIndex;
  support::ulittle16_t Type;
};

struct coff_aux_function_definition {
  support::ulittle32_t TagIndex;
  support::ulittle32_t TotalSize;
  support::ulittle32_t PointerToLinenumber;
  support::ulittle32_t PointerToNextFunction;
};

struct coff_aux_bf_and_ef_symbol {
  char Unused1[4];
  support::ulittle16_t Linenumber;
  char Unused2[6];
  support::ulittle32_t PointerToNextFunction;
};

struct coff_aux_weak_external {
  support::ulittle32_t TagIndex;
  support::ulittle32_t Characteristics;
};

struct coff_aux_section_definition {
  support::ulittle32_t Length;
  support::ulittle16_t NumberOfRelocations;
  support::ulittle16_t NumberOfLinenumbers;
  support::ulittle32_t CheckSum;
  support::ulittle16_t NumberLowPart;
  uint8_t              Selection;
  uint8_t              Unused;
  support::ulittle16_t NumberHighPart;
  int32_t getNumber(bool IsBigObj) const {
    uint32_t Number = static_cast<uint32_t>(NumberLowPart);
    if (IsBigObj)
      Number |= static_cast<uint32_t>(NumberHighPart) << 16;
    return static_cast<int32_t>(Number);
  }
};

struct coff_aux_clr_token {
  uint8_t              AuxType;
  uint8_t              Reserved;
  support::ulittle32_t SymbolTableIndex;
};

struct coff_import_header {
  support::ulittle16_t Sig1;
  support::ulittle16_t Sig2;
  support::ulittle16_t Version;
  support::ulittle16_t Machine;
  support::ulittle32_t TimeDateStamp;
  support::ulittle32_t SizeOfData;
  support::ulittle16_t OrdinalHint;
  support::ulittle16_t TypeInfo;
  int getType() const { return TypeInfo & 0x3; }
  int getNameType() const { return (TypeInfo >> 2) & 0x7; }
};

struct coff_import_directory_table_entry {
  support::ulittle32_t ImportLookupTableRVA;
  support::ulittle32_t TimeDateStamp;
  support::ulittle32_t ForwarderChain;
  support::ulittle32_t NameRVA;
  support::ulittle32_t ImportAddressTableRVA;
};

struct coff_load_configuration32 {
  support::ulittle32_t Characteristics;
  support::ulittle32_t TimeDateStamp;
  support::ulittle16_t MajorVersion;
  support::ulittle16_t MinorVersion;
  support::ulittle32_t GlobalFlagsClear;
  support::ulittle32_t GlobalFlagsSet;
  support::ulittle32_t CriticalSectionDefaultTimeout;
  support::ulittle32_t DeCommitFreeBlockThreshold;
  support::ulittle32_t DeCommitTotalFreeThreshold;
  support::ulittle32_t LockPrefixTable;
  support::ulittle32_t MaximumAllocationSize;
  support::ulittle32_t VirtualMemoryThreshold;
  support::ulittle32_t ProcessAffinityMask;
  support::ulittle32_t ProcessHeapFlags;
  support::ulittle16_t CSDVersion;
  support::ulittle16_t Reserved;
  support::ulittle32_t EditList;
  support::ulittle32_t SecurityCookie;
  support::ulittle32_t SEHandlerTable;
  support::ulittle32_t SEHandlerCount;
};

struct coff_load_configuration64 {
  support::ulittle32_t Characteristics;
  support::ulittle32_t TimeDateStamp;
  support::ulittle16_t MajorVersion;
  support::ulittle16_t MinorVersion;
  support::ulittle32_t GlobalFlagsClear;
  support::ulittle32_t GlobalFlagsSet;
  support::ulittle32_t CriticalSectionDefaultTimeout;
  support::ulittle32_t DeCommitFreeBlockThreshold;
  support::ulittle32_t DeCommitTotalFreeThreshold;
  support::ulittle32_t LockPrefixTable;
  support::ulittle32_t MaximumAllocationSize;
  support::ulittle32_t VirtualMemoryThreshold;
  support::ulittle32_t ProcessAffinityMask;
  support::ulittle32_t ProcessHeapFlags;
  support::ulittle16_t CSDVersion;
  support::ulittle16_t Reserved;
  support::ulittle32_t EditList;
  support::ulittle64_t SecurityCookie;
  support::ulittle64_t SEHandlerTable;
  support::ulittle64_t SEHandlerCount;
};

struct coff_runtime_function_x64 {
  support::ulittle32_t BeginAddress;
  support::ulittle32_t EndAddress;
  support::ulittle32_t UnwindInformation;
};

struct coff_base_reloc_block_header {
  support::ulittle32_t PageRVA;
  support::ulittle32_t BlockSize;
};

struct coff_base_reloc_block_entry {
  support::ulittle16_t Data;
  int getType() const { return Data >> 12; }
  int getOffset() const { return Data & ((1 << 12) - 1); }
};

class COFFObjectFile : public ObjectFile {
private:
  friend class ImportDirectoryEntryRef;
  friend class ExportDirectoryEntryRef;
  const coff_file_header *COFFHeader;
  const coff_bigobj_file_header *COFFBigObjHeader;
  const pe32_header *PE32Header;
  const pe32plus_header *PE32PlusHeader;
  const data_directory *DataDirectory;
  const coff_section *SectionTable;
  const coff_symbol16 *SymbolTable16;
  const coff_symbol32 *SymbolTable32;
  const char *StringTable;
  uint32_t StringTableSize;
  const import_directory_table_entry *ImportDirectory;
  uint32_t NumberOfImportDirectory;
  const delay_import_directory_table_entry *DelayImportDirectory;
  uint32_t NumberOfDelayImportDirectory;
  const export_directory_table_entry *ExportDirectory;
  const coff_base_reloc_block_header *BaseRelocHeader;
  const coff_base_reloc_block_header *BaseRelocEnd;

  std::error_code getString(uint32_t offset, StringRef &Res) const;

  template <typename coff_symbol_type>
  const coff_symbol_type *toSymb(DataRefImpl Symb) const;
  const coff_section *toSec(DataRefImpl Sec) const;
  const coff_relocation *toRel(DataRefImpl Rel) const;

  std::error_code initSymbolTablePtr();
  std::error_code initImportTablePtr();
  std::error_code initDelayImportTablePtr();
  std::error_code initExportTablePtr();
  std::error_code initBaseRelocPtr();

public:
  uintptr_t getSymbolTable() const {
    if (SymbolTable16)
      return reinterpret_cast<uintptr_t>(SymbolTable16);
    if (SymbolTable32)
      return reinterpret_cast<uintptr_t>(SymbolTable32);
    return uintptr_t(0);
  }
  uint16_t getMachine() const {
    if (COFFHeader)
      return COFFHeader->Machine;
    if (COFFBigObjHeader)
      return COFFBigObjHeader->Machine;
    llvm_unreachable("no COFF header!");
  }
  uint16_t getSizeOfOptionalHeader() const {
    if (COFFHeader)
      return COFFHeader->isImportLibrary() ? 0
                                           : COFFHeader->SizeOfOptionalHeader;
    // bigobj doesn't have this field.
    if (COFFBigObjHeader)
      return 0;
    llvm_unreachable("no COFF header!");
  }
  uint16_t getCharacteristics() const {
    if (COFFHeader)
      return COFFHeader->isImportLibrary() ? 0 : COFFHeader->Characteristics;
    // bigobj doesn't have characteristics to speak of,
    // editbin will silently lie to you if you attempt to set any.
    if (COFFBigObjHeader)
      return 0;
    llvm_unreachable("no COFF header!");
  }
  uint32_t getTimeDateStamp() const {
    if (COFFHeader)
      return COFFHeader->TimeDateStamp;
    if (COFFBigObjHeader)
      return COFFBigObjHeader->TimeDateStamp;
    llvm_unreachable("no COFF header!");
  }
  uint32_t getNumberOfSections() const {
    if (COFFHeader)
      return COFFHeader->isImportLibrary() ? 0 : COFFHeader->NumberOfSections;
    if (COFFBigObjHeader)
      return COFFBigObjHeader->NumberOfSections;
    llvm_unreachable("no COFF header!");
  }
  uint32_t getPointerToSymbolTable() const {
    if (COFFHeader)
      return COFFHeader->isImportLibrary() ? 0
                                           : COFFHeader->PointerToSymbolTable;
    if (COFFBigObjHeader)
      return COFFBigObjHeader->PointerToSymbolTable;
    llvm_unreachable("no COFF header!");
  }
  uint32_t getNumberOfSymbols() const {
    if (COFFHeader)
      return COFFHeader->isImportLibrary() ? 0 : COFFHeader->NumberOfSymbols;
    if (COFFBigObjHeader)
      return COFFBigObjHeader->NumberOfSymbols;
    llvm_unreachable("no COFF header!");
  }
protected:
  void moveSymbolNext(DataRefImpl &Symb) const override;
  ErrorOr<StringRef> getSymbolName(DataRefImpl Symb) const override;
  ErrorOr<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  uint32_t getSymbolFlags(DataRefImpl Symb) const override;
  SymbolRef::Type getSymbolType(DataRefImpl Symb) const override;
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

  void moveRelocationNext(DataRefImpl &Rel) const override;
  uint64_t getRelocationOffset(DataRefImpl Rel) const override;
  symbol_iterator getRelocationSymbol(DataRefImpl Rel) const override;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;

public:
  COFFObjectFile(MemoryBufferRef Object, std::error_code &EC);
  basic_symbol_iterator symbol_begin_impl() const override;
  basic_symbol_iterator symbol_end_impl() const override;
  section_iterator section_begin() const override;
  section_iterator section_end() const override;

  const coff_section *getCOFFSection(const SectionRef &Section) const;
  COFFSymbolRef getCOFFSymbol(const DataRefImpl &Ref) const;
  COFFSymbolRef getCOFFSymbol(const SymbolRef &Symbol) const;
  const coff_relocation *getCOFFRelocation(const RelocationRef &Reloc) const;
  unsigned getSectionID(SectionRef Sec) const;
  unsigned getSymbolSectionID(SymbolRef Sym) const;

  uint8_t getBytesInAddress() const override;
  StringRef getFileFormatName() const override;
  unsigned getArch() const override;

  import_directory_iterator import_directory_begin() const;
  import_directory_iterator import_directory_end() const;
  delay_import_directory_iterator delay_import_directory_begin() const;
  delay_import_directory_iterator delay_import_directory_end() const;
  export_directory_iterator export_directory_begin() const;
  export_directory_iterator export_directory_end() const;
  base_reloc_iterator base_reloc_begin() const;
  base_reloc_iterator base_reloc_end() const;

  iterator_range<import_directory_iterator> import_directories() const;
  iterator_range<delay_import_directory_iterator>
      delay_import_directories() const;
  iterator_range<export_directory_iterator> export_directories() const;
  iterator_range<base_reloc_iterator> base_relocs() const;

  const dos_header *getDOSHeader() const {
    if (!PE32Header && !PE32PlusHeader)
      return nullptr;
    return reinterpret_cast<const dos_header *>(base());
  }
  std::error_code getPE32Header(const pe32_header *&Res) const;
  std::error_code getPE32PlusHeader(const pe32plus_header *&Res) const;
  std::error_code getDataDirectory(uint32_t index,
                                   const data_directory *&Res) const;
  std::error_code getSection(int32_t index, const coff_section *&Res) const;
  template <typename coff_symbol_type>
  std::error_code getSymbol(uint32_t Index,
                            const coff_symbol_type *&Res) const {
    if (Index >= getNumberOfSymbols())
      return object_error::parse_failed;

    Res = reinterpret_cast<coff_symbol_type *>(getSymbolTable()) + Index;
    return std::error_code();
  }
  ErrorOr<COFFSymbolRef> getSymbol(uint32_t index) const {
    if (SymbolTable16) {
      const coff_symbol16 *Symb = nullptr;
      if (std::error_code EC = getSymbol(index, Symb))
        return EC;
      return COFFSymbolRef(Symb);
    }
    if (SymbolTable32) {
      const coff_symbol32 *Symb = nullptr;
      if (std::error_code EC = getSymbol(index, Symb))
        return EC;
      return COFFSymbolRef(Symb);
    }
    return object_error::parse_failed;
  }
  template <typename T>
  std::error_code getAuxSymbol(uint32_t index, const T *&Res) const {
    ErrorOr<COFFSymbolRef> s = getSymbol(index);
    if (std::error_code EC = s.getError())
      return EC;
    Res = reinterpret_cast<const T *>(s->getRawPtr());
    return std::error_code();
  }
  std::error_code getSymbolName(COFFSymbolRef Symbol, StringRef &Res) const;
  std::error_code getSymbolName(const coff_symbol_generic *Symbol,
                                StringRef &Res) const;

  ArrayRef<uint8_t> getSymbolAuxData(COFFSymbolRef Symbol) const;

  size_t getSymbolTableEntrySize() const {
    if (COFFHeader)
      return sizeof(coff_symbol16);
    if (COFFBigObjHeader)
      return sizeof(coff_symbol32);
    llvm_unreachable("null symbol table pointer!");
  }

  iterator_range<const coff_relocation *>
  getRelocations(const coff_section *Sec) const;

  std::error_code getSectionName(const coff_section *Sec, StringRef &Res) const;
  uint64_t getSectionSize(const coff_section *Sec) const;
  std::error_code getSectionContents(const coff_section *Sec,
                                     ArrayRef<uint8_t> &Res) const;

  std::error_code getVaPtr(uint64_t VA, uintptr_t &Res) const;
  std::error_code getRvaPtr(uint32_t Rva, uintptr_t &Res) const;
  std::error_code getHintName(uint32_t Rva, uint16_t &Hint,
                              StringRef &Name) const;

  bool isRelocatableObject() const override;
  bool is64() const { return PE32PlusHeader; }

  static inline bool classof(const Binary *v) { return v->isCOFF(); }
};

// The iterator for the import directory table.
class ImportDirectoryEntryRef {
public:
  ImportDirectoryEntryRef() : OwningObject(nullptr) {}
  ImportDirectoryEntryRef(const import_directory_table_entry *Table, uint32_t I,
                          const COFFObjectFile *Owner)
      : ImportTable(Table), Index(I), OwningObject(Owner) {}

  bool operator==(const ImportDirectoryEntryRef &Other) const;
  void moveNext();

  imported_symbol_iterator imported_symbol_begin() const;
  imported_symbol_iterator imported_symbol_end() const;
  iterator_range<imported_symbol_iterator> imported_symbols() const;

  std::error_code getName(StringRef &Result) const;
  std::error_code getImportLookupTableRVA(uint32_t &Result) const;
  std::error_code getImportAddressTableRVA(uint32_t &Result) const;

  std::error_code
  getImportTableEntry(const import_directory_table_entry *&Result) const;

  std::error_code
  getImportLookupEntry(const import_lookup_table_entry32 *&Result) const;

private:
  const import_directory_table_entry *ImportTable;
  uint32_t Index;
  const COFFObjectFile *OwningObject;
};

class DelayImportDirectoryEntryRef {
public:
  DelayImportDirectoryEntryRef() : OwningObject(nullptr) {}
  DelayImportDirectoryEntryRef(const delay_import_directory_table_entry *T,
                               uint32_t I, const COFFObjectFile *Owner)
      : Table(T), Index(I), OwningObject(Owner) {}

  bool operator==(const DelayImportDirectoryEntryRef &Other) const;
  void moveNext();

  imported_symbol_iterator imported_symbol_begin() const;
  imported_symbol_iterator imported_symbol_end() const;
  iterator_range<imported_symbol_iterator> imported_symbols() const;

  std::error_code getName(StringRef &Result) const;
  std::error_code getDelayImportTable(
      const delay_import_directory_table_entry *&Result) const;
  std::error_code getImportAddress(int AddrIndex, uint64_t &Result) const;

private:
  const delay_import_directory_table_entry *Table;
  uint32_t Index;
  const COFFObjectFile *OwningObject;
};

// The iterator for the export directory table entry.
class ExportDirectoryEntryRef {
public:
  ExportDirectoryEntryRef() : OwningObject(nullptr) {}
  ExportDirectoryEntryRef(const export_directory_table_entry *Table, uint32_t I,
                          const COFFObjectFile *Owner)
      : ExportTable(Table), Index(I), OwningObject(Owner) {}

  bool operator==(const ExportDirectoryEntryRef &Other) const;
  void moveNext();

  std::error_code getDllName(StringRef &Result) const;
  std::error_code getOrdinalBase(uint32_t &Result) const;
  std::error_code getOrdinal(uint32_t &Result) const;
  std::error_code getExportRVA(uint32_t &Result) const;
  std::error_code getSymbolName(StringRef &Result) const;

private:
  const export_directory_table_entry *ExportTable;
  uint32_t Index;
  const COFFObjectFile *OwningObject;
};

class ImportedSymbolRef {
public:
  ImportedSymbolRef() : OwningObject(nullptr) {}
  ImportedSymbolRef(const import_lookup_table_entry32 *Entry, uint32_t I,
                    const COFFObjectFile *Owner)
      : Entry32(Entry), Entry64(nullptr), Index(I), OwningObject(Owner) {}
  ImportedSymbolRef(const import_lookup_table_entry64 *Entry, uint32_t I,
                    const COFFObjectFile *Owner)
      : Entry32(nullptr), Entry64(Entry), Index(I), OwningObject(Owner) {}

  bool operator==(const ImportedSymbolRef &Other) const;
  void moveNext();

  std::error_code getSymbolName(StringRef &Result) const;
  std::error_code getOrdinal(uint16_t &Result) const;

private:
  const import_lookup_table_entry32 *Entry32;
  const import_lookup_table_entry64 *Entry64;
  uint32_t Index;
  const COFFObjectFile *OwningObject;
};

class BaseRelocRef {
public:
  BaseRelocRef() : OwningObject(nullptr) {}
  BaseRelocRef(const coff_base_reloc_block_header *Header,
               const COFFObjectFile *Owner)
      : Header(Header), Index(0), OwningObject(Owner) {}

  bool operator==(const BaseRelocRef &Other) const;
  void moveNext();

  std::error_code getType(uint8_t &Type) const;
  std::error_code getRVA(uint32_t &Result) const;

private:
  const coff_base_reloc_block_header *Header;
  uint32_t Index;
  const COFFObjectFile *OwningObject;
};

} // end namespace object
} // end namespace llvm

#endif
