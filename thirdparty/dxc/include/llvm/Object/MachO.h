//===- MachO.h - MachO object file implementation ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachOObjectFile class, which implement the ObjectFile
// interface for MachO files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHO_H
#define LLVM_OBJECT_MACHO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MachO.h"

namespace llvm {
namespace object {

/// DiceRef - This is a value type class that represents a single
/// data in code entry in the table in a Mach-O object file.
class DiceRef {
  DataRefImpl DicePimpl;
  const ObjectFile *OwningObject;

public:
  DiceRef() : OwningObject(nullptr) { }

  DiceRef(DataRefImpl DiceP, const ObjectFile *Owner);

  bool operator==(const DiceRef &Other) const;
  bool operator<(const DiceRef &Other) const;

  void moveNext();

  std::error_code getOffset(uint32_t &Result) const;
  std::error_code getLength(uint16_t &Result) const;
  std::error_code getKind(uint16_t &Result) const;

  DataRefImpl getRawDataRefImpl() const;
  const ObjectFile *getObjectFile() const;
};
typedef content_iterator<DiceRef> dice_iterator;

/// ExportEntry encapsulates the current-state-of-the-walk used when doing a
/// non-recursive walk of the trie data structure.  This allows you to iterate
/// across all exported symbols using:
///      for (const llvm::object::ExportEntry &AnExport : Obj->exports()) {
///      }
class ExportEntry {
public:
  ExportEntry(ArrayRef<uint8_t> Trie);

  StringRef name() const;
  uint64_t flags() const;
  uint64_t address() const;
  uint64_t other() const;
  StringRef otherName() const;
  uint32_t nodeOffset() const;

  bool operator==(const ExportEntry &) const;

  void moveNext();

private:
  friend class MachOObjectFile;
  void moveToFirst();
  void moveToEnd();
  uint64_t readULEB128(const uint8_t *&p);
  void pushDownUntilBottom();
  void pushNode(uint64_t Offset);

  // Represents a node in the mach-o exports trie.
  struct NodeState {
    NodeState(const uint8_t *Ptr);
    const uint8_t *Start;
    const uint8_t *Current;
    uint64_t Flags;
    uint64_t Address;
    uint64_t Other;
    const char *ImportName;
    unsigned ChildCount;
    unsigned NextChildIndex;
    unsigned ParentStringLength;
    bool IsExportNode;
  };

  ArrayRef<uint8_t> Trie;
  SmallString<256> CumulativeString;
  SmallVector<NodeState, 16> Stack;
  bool Malformed;
  bool Done;
};
typedef content_iterator<ExportEntry> export_iterator;

/// MachORebaseEntry encapsulates the current state in the decompression of   
/// rebasing opcodes. This allows you to iterate through the compressed table of
/// rebasing using:
///    for (const llvm::object::MachORebaseEntry &Entry : Obj->rebaseTable()) {
///    }
class MachORebaseEntry {
public:
  MachORebaseEntry(ArrayRef<uint8_t> opcodes, bool is64Bit);

  uint32_t segmentIndex() const;
  uint64_t segmentOffset() const;
  StringRef typeName() const;

  bool operator==(const MachORebaseEntry &) const;

  void moveNext();
  
private:
  friend class MachOObjectFile;
  void moveToFirst();
  void moveToEnd();
  uint64_t readULEB128();

  ArrayRef<uint8_t> Opcodes;
  const uint8_t *Ptr;
  uint64_t SegmentOffset;
  uint32_t SegmentIndex;
  uint64_t RemainingLoopCount;
  uint64_t AdvanceAmount;
  uint8_t  RebaseType;
  uint8_t  PointerSize;
  bool     Malformed;
  bool     Done;
};
typedef content_iterator<MachORebaseEntry> rebase_iterator;

/// MachOBindEntry encapsulates the current state in the decompression of
/// binding opcodes. This allows you to iterate through the compressed table of
/// bindings using:
///    for (const llvm::object::MachOBindEntry &Entry : Obj->bindTable()) {
///    }
class MachOBindEntry {
public:
  enum class Kind { Regular, Lazy, Weak };

  MachOBindEntry(ArrayRef<uint8_t> Opcodes, bool is64Bit, MachOBindEntry::Kind);

  uint32_t segmentIndex() const;
  uint64_t segmentOffset() const;
  StringRef typeName() const;
  StringRef symbolName() const;
  uint32_t flags() const;
  int64_t addend() const;
  int ordinal() const;

  bool operator==(const MachOBindEntry &) const;

  void moveNext();

private:
  friend class MachOObjectFile;
  void moveToFirst();
  void moveToEnd();
  uint64_t readULEB128();
  int64_t readSLEB128();

  ArrayRef<uint8_t> Opcodes;
  const uint8_t *Ptr;
  uint64_t SegmentOffset;
  uint32_t SegmentIndex;
  StringRef SymbolName;
  int      Ordinal;
  uint32_t Flags;
  int64_t  Addend;
  uint64_t RemainingLoopCount;
  uint64_t AdvanceAmount;
  uint8_t  BindType;
  uint8_t  PointerSize;
  Kind     TableKind;
  bool     Malformed;
  bool     Done;
};
typedef content_iterator<MachOBindEntry> bind_iterator;

class MachOObjectFile : public ObjectFile {
public:
  struct LoadCommandInfo {
    const char *Ptr;      // Where in memory the load command is.
    MachO::load_command C; // The command itself.
  };
  typedef SmallVector<LoadCommandInfo, 4> LoadCommandList;
  typedef LoadCommandList::const_iterator load_command_iterator;

  MachOObjectFile(MemoryBufferRef Object, bool IsLittleEndian, bool Is64Bits,
                  std::error_code &EC);

  void moveSymbolNext(DataRefImpl &Symb) const override;

  uint64_t getNValue(DataRefImpl Sym) const;
  ErrorOr<StringRef> getSymbolName(DataRefImpl Symb) const override;

  // MachO specific.
  std::error_code getIndirectName(DataRefImpl Symb, StringRef &Res) const;
  unsigned getSectionType(SectionRef Sec) const;

  ErrorOr<uint64_t> getSymbolAddress(DataRefImpl Symb) const override;
  uint32_t getSymbolAlignment(DataRefImpl Symb) const override;
  uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const override;
  SymbolRef::Type getSymbolType(DataRefImpl Symb) const override;
  uint32_t getSymbolFlags(DataRefImpl Symb) const override;
  std::error_code getSymbolSection(DataRefImpl Symb,
                                   section_iterator &Res) const override;
  unsigned getSymbolSectionID(SymbolRef Symb) const;
  unsigned getSectionID(SectionRef Sec) const;

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
  section_iterator getRelocationSection(DataRefImpl Rel) const;
  uint64_t getRelocationType(DataRefImpl Rel) const override;
  void getRelocationTypeName(DataRefImpl Rel,
                             SmallVectorImpl<char> &Result) const override;
  uint8_t getRelocationLength(DataRefImpl Rel) const;

  // MachO specific.
  std::error_code getLibraryShortNameByIndex(unsigned Index, StringRef &) const;

  section_iterator getRelocationRelocatedSection(relocation_iterator Rel) const;

  // TODO: Would be useful to have an iterator based version
  // of the load command interface too.

  basic_symbol_iterator symbol_begin_impl() const override;
  basic_symbol_iterator symbol_end_impl() const override;

  // MachO specific.
  basic_symbol_iterator getSymbolByIndex(unsigned Index) const;

  section_iterator section_begin() const override;
  section_iterator section_end() const override;

  uint8_t getBytesInAddress() const override;

  StringRef getFileFormatName() const override;
  unsigned getArch() const override;
  Triple getArch(const char **McpuDefault, Triple *ThumbTriple) const;

  relocation_iterator section_rel_begin(unsigned Index) const;
  relocation_iterator section_rel_end(unsigned Index) const;

  dice_iterator begin_dices() const;
  dice_iterator end_dices() const;

  load_command_iterator begin_load_commands() const;
  load_command_iterator end_load_commands() const;
  iterator_range<load_command_iterator> load_commands() const;

  /// For use iterating over all exported symbols.
  iterator_range<export_iterator> exports() const;

  /// For use examining a trie not in a MachOObjectFile.
  static iterator_range<export_iterator> exports(ArrayRef<uint8_t> Trie);

  /// For use iterating over all rebase table entries.
  iterator_range<rebase_iterator> rebaseTable() const;

  /// For use examining rebase opcodes not in a MachOObjectFile.
  static iterator_range<rebase_iterator> rebaseTable(ArrayRef<uint8_t> Opcodes,
                                                     bool is64);

  /// For use iterating over all bind table entries.
  iterator_range<bind_iterator> bindTable() const;

  /// For use iterating over all lazy bind table entries.
  iterator_range<bind_iterator> lazyBindTable() const;

  /// For use iterating over all lazy bind table entries.
  iterator_range<bind_iterator> weakBindTable() const;

  /// For use examining bind opcodes not in a MachOObjectFile.
  static iterator_range<bind_iterator> bindTable(ArrayRef<uint8_t> Opcodes,
                                                 bool is64,
                                                 MachOBindEntry::Kind);


  // In a MachO file, sections have a segment name. This is used in the .o
  // files. They have a single segment, but this field specifies which segment
  // a section should be put in in the final object.
  StringRef getSectionFinalSegmentName(DataRefImpl Sec) const;

  // Names are stored as 16 bytes. These returns the raw 16 bytes without
  // interpreting them as a C string.
  ArrayRef<char> getSectionRawName(DataRefImpl Sec) const;
  ArrayRef<char> getSectionRawFinalSegmentName(DataRefImpl Sec) const;

  // MachO specific Info about relocations.
  bool isRelocationScattered(const MachO::any_relocation_info &RE) const;
  unsigned getPlainRelocationSymbolNum(
                                    const MachO::any_relocation_info &RE) const;
  bool getPlainRelocationExternal(const MachO::any_relocation_info &RE) const;
  bool getScatteredRelocationScattered(
                                    const MachO::any_relocation_info &RE) const;
  uint32_t getScatteredRelocationValue(
                                    const MachO::any_relocation_info &RE) const;
  uint32_t getScatteredRelocationType(
                                    const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationAddress(const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationPCRel(const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationLength(const MachO::any_relocation_info &RE) const;
  unsigned getAnyRelocationType(const MachO::any_relocation_info &RE) const;
  SectionRef getAnyRelocationSection(const MachO::any_relocation_info &RE) const;

  // MachO specific structures.
  MachO::section getSection(DataRefImpl DRI) const;
  MachO::section_64 getSection64(DataRefImpl DRI) const;
  MachO::section getSection(const LoadCommandInfo &L, unsigned Index) const;
  MachO::section_64 getSection64(const LoadCommandInfo &L,unsigned Index) const;
  MachO::nlist getSymbolTableEntry(DataRefImpl DRI) const;
  MachO::nlist_64 getSymbol64TableEntry(DataRefImpl DRI) const;

  MachO::linkedit_data_command
  getLinkeditDataLoadCommand(const LoadCommandInfo &L) const;
  MachO::segment_command
  getSegmentLoadCommand(const LoadCommandInfo &L) const;
  MachO::segment_command_64
  getSegment64LoadCommand(const LoadCommandInfo &L) const;
  MachO::linker_option_command
  getLinkerOptionLoadCommand(const LoadCommandInfo &L) const;
  MachO::version_min_command
  getVersionMinLoadCommand(const LoadCommandInfo &L) const;
  MachO::dylib_command
  getDylibIDLoadCommand(const LoadCommandInfo &L) const;
  MachO::dyld_info_command
  getDyldInfoLoadCommand(const LoadCommandInfo &L) const;
  MachO::dylinker_command
  getDylinkerCommand(const LoadCommandInfo &L) const;
  MachO::uuid_command
  getUuidCommand(const LoadCommandInfo &L) const;
  MachO::rpath_command
  getRpathCommand(const LoadCommandInfo &L) const;
  MachO::source_version_command
  getSourceVersionCommand(const LoadCommandInfo &L) const;
  MachO::entry_point_command
  getEntryPointCommand(const LoadCommandInfo &L) const;
  MachO::encryption_info_command
  getEncryptionInfoCommand(const LoadCommandInfo &L) const;
  MachO::encryption_info_command_64
  getEncryptionInfoCommand64(const LoadCommandInfo &L) const;
  MachO::sub_framework_command
  getSubFrameworkCommand(const LoadCommandInfo &L) const;
  MachO::sub_umbrella_command
  getSubUmbrellaCommand(const LoadCommandInfo &L) const;
  MachO::sub_library_command
  getSubLibraryCommand(const LoadCommandInfo &L) const;
  MachO::sub_client_command
  getSubClientCommand(const LoadCommandInfo &L) const;
  MachO::routines_command
  getRoutinesCommand(const LoadCommandInfo &L) const;
  MachO::routines_command_64
  getRoutinesCommand64(const LoadCommandInfo &L) const;
  MachO::thread_command
  getThreadCommand(const LoadCommandInfo &L) const;

  MachO::any_relocation_info getRelocation(DataRefImpl Rel) const;
  MachO::data_in_code_entry getDice(DataRefImpl Rel) const;
  const MachO::mach_header &getHeader() const;
  const MachO::mach_header_64 &getHeader64() const;
  uint32_t
  getIndirectSymbolTableEntry(const MachO::dysymtab_command &DLC,
                              unsigned Index) const;
  MachO::data_in_code_entry getDataInCodeTableEntry(uint32_t DataOffset,
                                                    unsigned Index) const;
  MachO::symtab_command getSymtabLoadCommand() const;
  MachO::dysymtab_command getDysymtabLoadCommand() const;
  MachO::linkedit_data_command getDataInCodeLoadCommand() const;
  MachO::linkedit_data_command getLinkOptHintsLoadCommand() const;
  ArrayRef<uint8_t> getDyldInfoRebaseOpcodes() const;
  ArrayRef<uint8_t> getDyldInfoBindOpcodes() const;
  ArrayRef<uint8_t> getDyldInfoWeakBindOpcodes() const;
  ArrayRef<uint8_t> getDyldInfoLazyBindOpcodes() const;
  ArrayRef<uint8_t> getDyldInfoExportsTrie() const;
  ArrayRef<uint8_t> getUuid() const;

  StringRef getStringTableData() const;
  bool is64Bit() const;
  void ReadULEB128s(uint64_t Index, SmallVectorImpl<uint64_t> &Out) const;

  static StringRef guessLibraryShortName(StringRef Name, bool &isFramework,
                                         StringRef &Suffix);

  static Triple::ArchType getArch(uint32_t CPUType);
  static Triple getArch(uint32_t CPUType, uint32_t CPUSubType,
                        const char **McpuDefault = nullptr);
  static Triple getThumbArch(uint32_t CPUType, uint32_t CPUSubType,
                             const char **McpuDefault = nullptr);
  static Triple getArch(uint32_t CPUType, uint32_t CPUSubType,
                        const char **McpuDefault, Triple *ThumbTriple);
  static bool isValidArch(StringRef ArchFlag);
  static Triple getHostArch();

  bool isRelocatableObject() const override;

  bool hasPageZeroSegment() const { return HasPageZeroSegment; }

  static bool classof(const Binary *v) {
    return v->isMachO();
  }

private:
  uint64_t getSymbolValueImpl(DataRefImpl Symb) const override;

  union {
    MachO::mach_header_64 Header64;
    MachO::mach_header Header;
  };
  typedef SmallVector<const char*, 1> SectionList;
  SectionList Sections;
  typedef SmallVector<const char*, 1> LibraryList;
  LibraryList Libraries;
  LoadCommandList LoadCommands;
  typedef SmallVector<StringRef, 1> LibraryShortName;
  mutable LibraryShortName LibrariesShortNames;
  const char *SymtabLoadCmd;
  const char *DysymtabLoadCmd;
  const char *DataInCodeLoadCmd;
  const char *LinkOptHintsLoadCmd;
  const char *DyldInfoLoadCmd;
  const char *UuidLoadCmd;
  bool HasPageZeroSegment;
};

/// DiceRef
inline DiceRef::DiceRef(DataRefImpl DiceP, const ObjectFile *Owner)
  : DicePimpl(DiceP) , OwningObject(Owner) {}

inline bool DiceRef::operator==(const DiceRef &Other) const {
  return DicePimpl == Other.DicePimpl;
}

inline bool DiceRef::operator<(const DiceRef &Other) const {
  return DicePimpl < Other.DicePimpl;
}

inline void DiceRef::moveNext() {
  const MachO::data_in_code_entry *P =
    reinterpret_cast<const MachO::data_in_code_entry *>(DicePimpl.p);
  DicePimpl.p = reinterpret_cast<uintptr_t>(P + 1);
}

// Since a Mach-O data in code reference, a DiceRef, can only be created when
// the OwningObject ObjectFile is a MachOObjectFile a static_cast<> is used for
// the methods that get the values of the fields of the reference.

inline std::error_code DiceRef::getOffset(uint32_t &Result) const {
  const MachOObjectFile *MachOOF =
    static_cast<const MachOObjectFile *>(OwningObject);
  MachO::data_in_code_entry Dice = MachOOF->getDice(DicePimpl);
  Result = Dice.offset;
  return std::error_code();
}

inline std::error_code DiceRef::getLength(uint16_t &Result) const {
  const MachOObjectFile *MachOOF =
    static_cast<const MachOObjectFile *>(OwningObject);
  MachO::data_in_code_entry Dice = MachOOF->getDice(DicePimpl);
  Result = Dice.length;
  return std::error_code();
}

inline std::error_code DiceRef::getKind(uint16_t &Result) const {
  const MachOObjectFile *MachOOF =
    static_cast<const MachOObjectFile *>(OwningObject);
  MachO::data_in_code_entry Dice = MachOOF->getDice(DicePimpl);
  Result = Dice.kind;
  return std::error_code();
}

inline DataRefImpl DiceRef::getRawDataRefImpl() const {
  return DicePimpl;
}

inline const ObjectFile *DiceRef::getObjectFile() const {
  return OwningObject;
}

}
}

#endif

