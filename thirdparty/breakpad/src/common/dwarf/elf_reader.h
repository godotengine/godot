// Copyright 2005 Google LLC
// Author: chatham@google.com (Andrew Chatham)
// Author: satorux@google.com (Satoru Takabayashi)
//
// ElfReader handles reading in ELF. It can extract symbols from the
// current process, which may be used to symbolize stack traces
// without having to make a potentially dangerous call to fork().
//
// ElfReader dynamically allocates memory, so it is not appropriate to
// use once the address space might be corrupted, such as during
// process death.
//
// ElfReader supports both 32-bit and 64-bit ELF binaries.

#ifndef COMMON_DWARF_ELF_READER_H__
#define COMMON_DWARF_ELF_READER_H__

#include <string>
#include <string_view>
#include <vector>

#include "common/dwarf/types.h"
#include "common/using_std_string.h"

using std::vector;
using std::pair;

namespace google_breakpad {

class SymbolMap;
class Elf32;
class Elf64;
template<typename ElfArch>
class ElfReaderImpl;

class ElfReader {
 public:
  explicit ElfReader(const string& path);
  ~ElfReader();

  // Parse the ELF prologue of this file and return whether it was
  // successfully parsed and matches the word size and byte order of
  // the current process.
  bool IsNativeElfFile() const;

  // Similar to IsNativeElfFile but checks if it's a 32-bit ELF file.
  bool IsElf32File() const;

  // Similar to IsNativeElfFile but checks if it's a 64-bit ELF file.
  bool IsElf64File() const;

  // Checks if it's an ELF file of type ET_DYN (shared object file).
  bool IsDynamicSharedObject();

  // Add symbols in the given ELF file into the provided SymbolMap,
  // assuming that the file has been loaded into the specified
  // offset.
  //
  // The remaining arguments are typically taken from a
  // ProcMapsIterator (base/sysinfo.h) and describe which portions of
  // the ELF file are mapped into which parts of memory:
  //
  // mem_offset - position at which the segment is mapped into memory
  // file_offset - offset in the file where the mapping begins
  // length - length of the mapped segment
  void AddSymbols(SymbolMap* symbols,
                  uint64_t mem_offset, uint64_t file_offset,
                  uint64_t length);

  class SymbolSink {
   public:
    virtual ~SymbolSink() {}
    virtual void AddSymbol(const char* name, uint64_t address,
                           uint64_t size) = 0;
  };

  // Like AddSymbols above, but with no address correction.
  // Processes any SHT_SYMTAB section, followed by any SHT_DYNSYM section.
  void VisitSymbols(SymbolSink* sink);

  // Like VisitSymbols above, but for a specific symbol binding/type.
  // A negative value for the binding and type parameters means any
  // binding or type.
  void VisitSymbols(SymbolSink* sink, int symbol_binding, int symbol_type);

  // Like VisitSymbols above but can optionally export raw symbol values instead
  // of adjusted ones.
  void VisitSymbols(SymbolSink* sink, int symbol_binding, int symbol_type,
                    bool get_raw_symbol_values);

  // p_vaddr of the first PT_LOAD segment (if any), or 0 if no PT_LOAD
  // segments are present. This is the address an ELF image was linked
  // (by static linker) to be loaded at. Usually (but not always) 0 for
  // shared libraries and position-independent executables.
  uint64_t VaddrOfFirstLoadSegment();

  // Return the name of section "shndx".  Returns NULL if the section
  // is not found.
  const char* GetSectionName(int shndx);

  // Return the number of sections in the given ELF file.
  uint64_t GetNumSections();

  // Get section "shndx" from the given ELF file.  On success, return
  // the pointer to the section and store the size in "size".
  // On error, return NULL.  The returned section data is only valid
  // until the ElfReader gets destroyed.
  const char* GetSectionByIndex(int shndx, size_t* size);

  // Get section with "section_name" (ex. ".text", ".symtab") in the
  // given ELF file.  On success, return the pointer to the section
  // and store the size in "size".  On error, return NULL.  The
  // returned section data is only valid until the ElfReader gets
  // destroyed.
  const char* GetSectionByName(const string& section_name, size_t* size);

  // This is like GetSectionByName() but it returns a lot of extra information
  // about the section. The SectionInfo structure is almost identical to
  // the typedef struct Elf64_Shdr defined in <elf.h>, but is redefined
  // here so that the many short macro names in <elf.h> don't have to be
  // added to our already cluttered namespace.
  struct SectionInfo {
    uint32_t type;              // Section type (SHT_xxx constant from elf.h).
    uint64_t flags;             // Section flags (SHF_xxx constants from elf.h).
    uint64_t addr;              // Section virtual address at execution.
    uint64_t offset;            // Section file offset.
    uint64_t size;              // Section size in bytes.
    uint32_t link;              // Link to another section.
    uint32_t info;              // Additional section information.
    uint64_t addralign;         // Section alignment.
    uint64_t entsize;           // Entry size if section holds a table.
  };
  const char* GetSectionInfoByName(const string& section_name,
                                   SectionInfo* info);

  // Check if "path" is an ELF binary that has not been stripped of symbol
  // tables.  This function supports both 32-bit and 64-bit ELF binaries.
  static bool IsNonStrippedELFBinary(const string& path);

  // Check if "path" is an ELF binary that has not been stripped of debug
  // info. Unlike IsNonStrippedELFBinary, this function will return
  // false for binaries passed through "strip -S".
  static bool IsNonDebugStrippedELFBinary(const string& path);

  // Match a requested section name with the section name as it
  // appears in the elf-file, adjusting for compressed debug section
  // names.  For example, returns true if name == ".debug_abbrev" and
  // sh_name == ".zdebug_abbrev"
  static bool SectionNamesMatch(std::string_view name,
                                std::string_view sh_name);

 private:
  // Lazily initialize impl32_ and return it.
  ElfReaderImpl<Elf32>* GetImpl32();
  // Ditto for impl64_.
  ElfReaderImpl<Elf64>* GetImpl64();

  // Path of the file we're reading.
  const string path_;
  // Read-only file descriptor for the file. May be -1 if there was an
  // error during open.
  int fd_;
  ElfReaderImpl<Elf32>* impl32_;
  ElfReaderImpl<Elf64>* impl64_;
};

}  // namespace google_breakpad

#endif  // COMMON_DWARF_ELF_READER_H__
