// Copyright 2011 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Restructured in 2009 by: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// dump_symbols.cc: implement google_breakpad::WriteSymbolFile:
// Find all the debugging info in a file and dump it as a Breakpad symbol file.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/linux/dump_symbols.h"

#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>
#ifdef HAVE_LIBZSTD
#include <zstd.h>
#endif

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/dwarf/bytereader-inl.h"
#include "common/dwarf/dwarf2diehandler.h"
#include "common/dwarf_cfi_to_module.h"
#include "common/dwarf_cu_to_module.h"
#include "common/dwarf_line_to_module.h"
#include "common/dwarf_range_list_handler.h"
#include "common/linux/crc32.h"
#include "common/linux/eintr_wrapper.h"
#include "common/linux/elfutils.h"
#include "common/linux/elfutils-inl.h"
#include "common/linux/elf_symbols_to_module.h"
#include "common/linux/file_id.h"
#include "common/memory_allocator.h"
#include "common/module.h"
#include "common/path_helper.h"
#include "common/scoped_ptr.h"
#ifndef NO_STABS_SUPPORT
#include "common/stabs_reader.h"
#include "common/stabs_to_module.h"
#endif
#include "common/using_std_string.h"

// This namespace contains helper functions.
namespace {

using google_breakpad::DumpOptions;
using google_breakpad::DwarfCFIToModule;
using google_breakpad::DwarfCUToModule;
using google_breakpad::DwarfLineToModule;
using google_breakpad::DwarfRangeListHandler;
using google_breakpad::ElfClass;
using google_breakpad::ElfClass32;
using google_breakpad::ElfClass64;
using google_breakpad::elf::FileID;
using google_breakpad::FindElfSectionByName;
using google_breakpad::GetOffset;
using google_breakpad::IsValidElf;
using google_breakpad::elf::kDefaultBuildIdSize;
using google_breakpad::Module;
using google_breakpad::PageAllocator;
#ifndef NO_STABS_SUPPORT
using google_breakpad::StabsToModule;
#endif
using google_breakpad::scoped_ptr;
using google_breakpad::wasteful_vector;

// Define AARCH64 ELF architecture if host machine does not include this define.
#ifndef EM_AARCH64
#define EM_AARCH64      183
#endif

// Define ZStd compression if host machine does not include this define.
#ifndef ELFCOMPRESS_ZSTD
#define ELFCOMPRESS_ZSTD 2
#endif

//
// FDWrapper
//
// Wrapper class to make sure opened file is closed.
//
class FDWrapper {
 public:
  explicit FDWrapper(int fd) :
    fd_(fd) {}
  ~FDWrapper() {
    if (fd_ != -1)
      close(fd_);
  }
  int get() {
    return fd_;
  }
  int release() {
    int fd = fd_;
    fd_ = -1;
    return fd;
  }
 private:
  int fd_;
};

//
// MmapWrapper
//
// Wrapper class to make sure mapped regions are unmapped.
//
class MmapWrapper {
 public:
  MmapWrapper() : is_set_(false) {}
  ~MmapWrapper() {
    if (is_set_ && base_ != NULL) {
      assert(size_ > 0);
      munmap(base_, size_);
    }
  }
  void set(void* mapped_address, size_t mapped_size) {
    is_set_ = true;
    base_ = mapped_address;
    size_ = mapped_size;
  }
  void release() {
    assert(is_set_);
    is_set_ = false;
    base_ = NULL;
    size_ = 0;
  }

 private:
  bool is_set_;
  void* base_;
  size_t size_;
};

// Find the preferred loading address of the binary.
template<typename ElfClass>
typename ElfClass::Addr GetLoadingAddress(
    const typename ElfClass::Phdr* program_headers,
    int nheader) {
  typedef typename ElfClass::Phdr Phdr;

  // For non-PIC executables (e_type == ET_EXEC), the load address is
  // the start address of the first PT_LOAD segment.  (ELF requires
  // the segments to be sorted by load address.)  For PIC executables
  // and dynamic libraries (e_type == ET_DYN), this address will
  // normally be zero.
  for (int i = 0; i < nheader; ++i) {
    const Phdr& header = program_headers[i];
    if (header.p_type == PT_LOAD)
      return header.p_vaddr;
  }
  return 0;
}

// Find the set of address ranges for all PT_LOAD segments.
template <typename ElfClass>
vector<Module::Range> GetPtLoadSegmentRanges(
    const typename ElfClass::Phdr* program_headers,
    int nheader) {
  typedef typename ElfClass::Phdr Phdr;
  vector<Module::Range> ranges;

  for (int i = 0; i < nheader; ++i) {
    const Phdr& header = program_headers[i];
    if (header.p_type == PT_LOAD) {
      ranges.push_back(Module::Range(header.p_vaddr, header.p_memsz));
    }
  }
  return ranges;
}

#ifndef NO_STABS_SUPPORT
template<typename ElfClass>
bool LoadStabs(const typename ElfClass::Ehdr* elf_header,
               const typename ElfClass::Shdr* stab_section,
               const typename ElfClass::Shdr* stabstr_section,
               const bool big_endian,
               Module* module) {
  // A callback object to handle data from the STABS reader.
  StabsToModule handler(module);
  // Find the addresses of the STABS data, and create a STABS reader object.
  // On Linux, STABS entries always have 32-bit values, regardless of the
  // address size of the architecture whose code they're describing, and
  // the strings are always "unitized".
  const uint8_t* stabs =
      GetOffset<ElfClass, uint8_t>(elf_header, stab_section->sh_offset);
  const uint8_t* stabstr =
      GetOffset<ElfClass, uint8_t>(elf_header, stabstr_section->sh_offset);
  google_breakpad::StabsReader reader(stabs, stab_section->sh_size,
                                      stabstr, stabstr_section->sh_size,
                                      big_endian, 4, true, &handler);
  // Read the STABS data, and do post-processing.
  if (!reader.Process())
    return false;
  handler.Finalize();
  return true;
}
#endif  // NO_STABS_SUPPORT

// A range handler that accepts rangelist data parsed by
// google_breakpad::RangeListReader and populates a range vector (typically
// owned by a function) with the results.
class DumperRangesHandler : public DwarfCUToModule::RangesHandler {
 public:
  DumperRangesHandler(google_breakpad::ByteReader* reader) :
      reader_(reader) { }

  bool ReadRanges(
      enum google_breakpad::DwarfForm form, uint64_t data,
      google_breakpad::RangeListReader::CURangesInfo* cu_info,
      vector<Module::Range>* ranges) {
    DwarfRangeListHandler handler(ranges);
    google_breakpad::RangeListReader range_list_reader(reader_, cu_info,
                                                    &handler);
    return range_list_reader.ReadRanges(form, data);
  }

 private:
  google_breakpad::ByteReader* reader_;
};

// A line-to-module loader that accepts line number info parsed by
// google_breakpad::LineInfo and populates a Module and a line vector
// with the results.
class DumperLineToModule: public DwarfCUToModule::LineToModuleHandler {
 public:
  // Create a line-to-module converter using BYTE_READER.
  explicit DumperLineToModule(google_breakpad::ByteReader* byte_reader)
      : byte_reader_(byte_reader) { }
  void StartCompilationUnit(const string& compilation_dir) {
    compilation_dir_ = compilation_dir;
  }
  void ReadProgram(const uint8_t* program,
                   uint64_t length,
                   const uint8_t* string_section,
                   uint64_t string_section_length,
                   const uint8_t* line_string_section,
                   uint64_t line_string_section_length,
                   Module* module,
                   std::vector<Module::Line>* lines,
                   std::map<uint32_t, Module::File*>* files) {
    DwarfLineToModule handler(module, compilation_dir_, lines, files);
    google_breakpad::LineInfo parser(program, length, byte_reader_,
                                  string_section, string_section_length,
                                  line_string_section,
                                  line_string_section_length,
                                  &handler);
    parser.Start();
  }
 private:
  string compilation_dir_;
  google_breakpad::ByteReader* byte_reader_;
};

template<typename ElfClass>
bool IsCompressedHeader(const typename ElfClass::Shdr* section) {
  return (section->sh_flags & SHF_COMPRESSED) != 0;
}

template<typename ElfClass>
uint32_t GetCompressionHeader(
    typename ElfClass::Chdr& compression_header,
    const uint8_t* content, uint64_t size) {
  const typename ElfClass::Chdr* header =
      reinterpret_cast<const typename ElfClass::Chdr *>(content);

  if (size < sizeof (*header)) {
    return 0;
  }

  compression_header = *header;
  return sizeof (*header);
}

std::pair<uint8_t *, uint64_t> UncompressZlibSectionContents(
    const uint8_t* compressed_buffer, uint64_t compressed_size, uint64_t uncompressed_size) {
  google_breakpad::scoped_array<uint8_t> uncompressed_buffer(
    new uint8_t[uncompressed_size]);

  uLongf size = static_cast<uLongf>(uncompressed_size);

  int status = uncompress(
    uncompressed_buffer.get(), &size, compressed_buffer, compressed_size);

  return status != Z_OK
      ? std::make_pair(nullptr, 0)
      : std::make_pair(uncompressed_buffer.release(), uncompressed_size);
}

#ifdef HAVE_LIBZSTD
std::pair<uint8_t *, uint64_t> UncompressZstdSectionContents(
    const uint8_t* compressed_buffer, uint64_t compressed_size,uint64_t uncompressed_size) {

  google_breakpad::scoped_array<uint8_t> uncompressed_buffer(new uint8_t[uncompressed_size]);
  size_t out_size = ZSTD_decompress(uncompressed_buffer.get(), uncompressed_size,
    compressed_buffer, compressed_size);
  if (ZSTD_isError(out_size)) {
    return std::make_pair(nullptr, 0);
  }
  assert(out_size == uncompressed_size);
  return std::make_pair(uncompressed_buffer.release(), uncompressed_size);
}
#endif

std::pair<uint8_t *, uint64_t> UncompressSectionContents(
    uint64_t compression_type, const uint8_t* compressed_buffer,
    uint64_t compressed_size, uint64_t uncompressed_size) {
  if (compression_type == ELFCOMPRESS_ZLIB) {
    return UncompressZlibSectionContents(compressed_buffer, compressed_size, uncompressed_size);
  }

#ifdef HAVE_LIBZSTD
  if (compression_type == ELFCOMPRESS_ZSTD) {
    return UncompressZstdSectionContents(compressed_buffer, compressed_size, uncompressed_size);
  }
#endif

  return std::make_pair(nullptr, 0);
}

void StartProcessSplitDwarf(google_breakpad::CompilationUnit* reader,
                            Module* module,
                            google_breakpad::Endianness endianness,
                            bool handle_inter_cu_refs,
                            bool handle_inline) {
  std::string split_file;
  google_breakpad::SectionMap split_sections;
  google_breakpad::ByteReader split_byte_reader(endianness);
  uint64_t cu_offset = 0;
  if (!reader->ProcessSplitDwarf(split_file, split_sections, split_byte_reader,
                                 cu_offset))
    return;
  DwarfCUToModule::FileContext file_context(split_file, module,
                                            handle_inter_cu_refs);
  for (auto section : split_sections)
    file_context.AddSectionToSectionMap(section.first, section.second.first,
                                        section.second.second);
  // Because DWP/DWO file doesn't have .debug_addr/.debug_line/.debug_line_str,
  // its debug info will refer to .debug_addr/.debug_line in the main binary.
  if (file_context.section_map().find(".debug_addr") ==
      file_context.section_map().end())
    file_context.AddSectionToSectionMap(".debug_addr", reader->GetAddrBuffer(),
                                        reader->GetAddrBufferLen());
  if (file_context.section_map().find(".debug_line") ==
      file_context.section_map().end())
    file_context.AddSectionToSectionMap(".debug_line", reader->GetLineBuffer(),
                                        reader->GetLineBufferLen());
  if (file_context.section_map().find(".debug_line_str") ==
      file_context.section_map().end())
    file_context.AddSectionToSectionMap(".debug_line_str",
                                        reader->GetLineStrBuffer(),
                                        reader->GetLineStrBufferLen());

  DumperRangesHandler ranges_handler(&split_byte_reader);
  DumperLineToModule line_to_module(&split_byte_reader);
  DwarfCUToModule::WarningReporter reporter(split_file, cu_offset);
  DwarfCUToModule root_handler(
      &file_context, &line_to_module, &ranges_handler, &reporter, handle_inline,
      reader->GetLowPC(), reader->GetAddrBase(), reader->HasSourceLineInfo(),
      reader->GetSourceLineOffset());
  google_breakpad::DIEDispatcher die_dispatcher(&root_handler);
  google_breakpad::CompilationUnit split_reader(
      split_file, file_context.section_map(), cu_offset, &split_byte_reader,
      &die_dispatcher);
  split_reader.SetSplitDwarf(reader->GetAddrBase(), reader->GetDWOID());
  split_reader.Start();
  // Normally, it won't happen unless we have transitive reference.
  if (split_reader.ShouldProcessSplitDwarf()) {
    StartProcessSplitDwarf(&split_reader, module, endianness,
                           handle_inter_cu_refs, handle_inline);
  }
}

template<typename ElfClass>
bool LoadDwarf(const string& dwarf_filename,
               const typename ElfClass::Ehdr* elf_header,
               const bool big_endian,
               bool handle_inter_cu_refs,
               bool handle_inline,
               Module* module) {
  typedef typename ElfClass::Shdr Shdr;

  const google_breakpad::Endianness endianness = big_endian ?
      google_breakpad::ENDIANNESS_BIG : google_breakpad::ENDIANNESS_LITTLE;
  google_breakpad::ByteReader byte_reader(endianness);

  // Construct a context for this file.
  DwarfCUToModule::FileContext file_context(dwarf_filename,
                                            module,
                                            handle_inter_cu_refs);

  // Build a map of the ELF file's sections.
  const Shdr* sections =
      GetOffset<ElfClass, Shdr>(elf_header, elf_header->e_shoff);
  int num_sections = elf_header->e_shnum;
  const Shdr* section_names = sections + elf_header->e_shstrndx;
  for (int i = 0; i < num_sections; i++) {
    const Shdr* section = &sections[i];
    string name = GetOffset<ElfClass, char>(elf_header,
                                            section_names->sh_offset) +
                  section->sh_name;
    const uint8_t* contents = GetOffset<ElfClass, uint8_t>(elf_header,
                                                           section->sh_offset);
    uint64_t size = section->sh_size;

    if (!IsCompressedHeader<ElfClass>(section)) {
      file_context.AddSectionToSectionMap(name, contents, size);
      continue;
    }

    typename ElfClass::Chdr chdr;

    uint32_t compression_header_size =
      GetCompressionHeader<ElfClass>(chdr, contents, size);

    if (compression_header_size == 0 || chdr.ch_size == 0) {
      continue;
    }

    contents += compression_header_size;
    size -= compression_header_size;

    std::pair<uint8_t *, uint64_t> uncompressed =
      UncompressSectionContents(chdr.ch_type, contents, size, chdr.ch_size);

    if (uncompressed.first != nullptr && uncompressed.second != 0) {
      file_context.AddManagedSectionToSectionMap(name, uncompressed.first, uncompressed.second);
    }
  }

  // .debug_ranges and .debug_rnglists reader
  DumperRangesHandler ranges_handler(&byte_reader);

  // Parse all the compilation units in the .debug_info section.
  DumperLineToModule line_to_module(&byte_reader);
  google_breakpad::SectionMap::const_iterator debug_info_entry =
      file_context.section_map().find(".debug_info");
  assert(debug_info_entry != file_context.section_map().end());
  const std::pair<const uint8_t*, uint64_t>& debug_info_section =
      debug_info_entry->second;
  // This should never have been called if the file doesn't have a
  // .debug_info section.
  assert(debug_info_section.first);
  uint64_t debug_info_length = debug_info_section.second;
  for (uint64_t offset = 0; offset < debug_info_length;) {
    // Make a handler for the root DIE that populates MODULE with the
    // data that was found.
    DwarfCUToModule::WarningReporter reporter(dwarf_filename, offset);
    DwarfCUToModule root_handler(&file_context, &line_to_module,
                                 &ranges_handler, &reporter, handle_inline);
    // Make a Dwarf2Handler that drives the DIEHandler.
    google_breakpad::DIEDispatcher die_dispatcher(&root_handler);
    // Make a DWARF parser for the compilation unit at OFFSET.
    google_breakpad::CompilationUnit reader(dwarf_filename,
                                         file_context.section_map(),
                                         offset,
                                         &byte_reader,
                                         &die_dispatcher);
    // Process the entire compilation unit; get the offset of the next.
    uint64_t result = reader.Start();
    if (result == 0) {
      return false;
    }
    offset += result;
    // Start to process split dwarf file.
    if (reader.ShouldProcessSplitDwarf()) {
      StartProcessSplitDwarf(&reader, module, endianness, handle_inter_cu_refs,
                             handle_inline);
    }
  }
  return true;
}

// Fill REGISTER_NAMES with the register names appropriate to the
// machine architecture given in HEADER, indexed by the register
// numbers used in DWARF call frame information. Return true on
// success, or false if HEADER's machine architecture is not
// supported.
template<typename ElfClass>
bool DwarfCFIRegisterNames(const typename ElfClass::Ehdr* elf_header,
                           std::vector<string>* register_names) {
  switch (elf_header->e_machine) {
    case EM_386:
      *register_names = DwarfCFIToModule::RegisterNames::I386();
      return true;
    case EM_ARM:
      *register_names = DwarfCFIToModule::RegisterNames::ARM();
      return true;
    case EM_AARCH64:
      *register_names = DwarfCFIToModule::RegisterNames::ARM64();
      return true;
    case EM_MIPS:
      *register_names = DwarfCFIToModule::RegisterNames::MIPS();
      return true;
    case EM_X86_64:
      *register_names = DwarfCFIToModule::RegisterNames::X86_64();
      return true;
    case EM_RISCV:
      *register_names = DwarfCFIToModule::RegisterNames::RISCV();
      return true;
    default:
      return false;
  }
}

template<typename ElfClass>
bool LoadDwarfCFI(const string& dwarf_filename,
                  const typename ElfClass::Ehdr* elf_header,
                  const char* section_name,
                  const typename ElfClass::Shdr* section,
                  const bool eh_frame,
                  const typename ElfClass::Shdr* got_section,
                  const typename ElfClass::Shdr* text_section,
                  const bool big_endian,
                  Module* module) {
  // Find the appropriate set of register names for this file's
  // architecture.
  std::vector<string> register_names;
  if (!DwarfCFIRegisterNames<ElfClass>(elf_header, &register_names)) {
    fprintf(stderr, "%s: unrecognized ELF machine architecture '%d';"
            " cannot convert DWARF call frame information\n",
            dwarf_filename.c_str(), elf_header->e_machine);
    return false;
  }

  const google_breakpad::Endianness endianness = big_endian ?
      google_breakpad::ENDIANNESS_BIG : google_breakpad::ENDIANNESS_LITTLE;

  // Find the call frame information and its size.
  const uint8_t* cfi =
      GetOffset<ElfClass, uint8_t>(elf_header, section->sh_offset);
  size_t cfi_size = section->sh_size;

  // Plug together the parser, handler, and their entourages.
  DwarfCFIToModule::Reporter module_reporter(dwarf_filename, section_name);
  DwarfCFIToModule handler(module, register_names, &module_reporter);
  google_breakpad::ByteReader byte_reader(endianness);

  byte_reader.SetAddressSize(ElfClass::kAddrSize);

  // Provide the base addresses for .eh_frame encoded pointers, if
  // possible.
  byte_reader.SetCFIDataBase(section->sh_addr, cfi);
  if (got_section)
    byte_reader.SetDataBase(got_section->sh_addr);
  if (text_section)
    byte_reader.SetTextBase(text_section->sh_addr);

  google_breakpad::CallFrameInfo::Reporter dwarf_reporter(dwarf_filename,
                                                       section_name);
  if (!IsCompressedHeader<ElfClass>(section)) {
    google_breakpad::CallFrameInfo parser(cfi, cfi_size,
                                          &byte_reader, &handler,
                                          &dwarf_reporter, eh_frame);
    parser.Start();
    return true;
  }

  typename ElfClass::Chdr chdr;
  uint32_t compression_header_size =
    GetCompressionHeader<ElfClass>(chdr, cfi, cfi_size);

  if (compression_header_size == 0 || chdr.ch_size == 0) {
    fprintf(stderr, "%s: decompression failed at header\n",
            dwarf_filename.c_str());
    return false;
  }
  if (compression_header_size > cfi_size) {
    fprintf(stderr, "%s: decompression error, compression_header too large\n",
            dwarf_filename.c_str());
    return false;
  }

  cfi += compression_header_size;
  cfi_size -= compression_header_size;

  std::pair<uint8_t *, uint64_t> uncompressed =
    UncompressSectionContents(chdr.ch_type, cfi, cfi_size, chdr.ch_size);

  if (uncompressed.first == nullptr || uncompressed.second == 0) {
    fprintf(stderr, "%s: decompression failed\n", dwarf_filename.c_str());
    return false;
  }
  google_breakpad::CallFrameInfo parser(uncompressed.first, uncompressed.second,
                                        &byte_reader, &handler, &dwarf_reporter,
                                        eh_frame);
  parser.Start();
  return true;
}

bool LoadELF(const string& obj_file, MmapWrapper* map_wrapper,
             void** elf_header) {
  int obj_fd = open(obj_file.c_str(), O_RDONLY);
  if (obj_fd < 0) {
    fprintf(stderr, "Failed to open ELF file '%s': %s\n",
            obj_file.c_str(), strerror(errno));
    return false;
  }
  FDWrapper obj_fd_wrapper(obj_fd);
  struct stat st;
  if (fstat(obj_fd, &st) != 0 && st.st_size <= 0) {
    fprintf(stderr, "Unable to fstat ELF file '%s': %s\n",
            obj_file.c_str(), strerror(errno));
    return false;
  }
  void* obj_base = mmap(NULL, st.st_size,
                        PROT_READ | PROT_WRITE, MAP_PRIVATE, obj_fd, 0);
  if (obj_base == MAP_FAILED) {
    fprintf(stderr, "Failed to mmap ELF file '%s': %s\n",
            obj_file.c_str(), strerror(errno));
    return false;
  }
  map_wrapper->set(obj_base, st.st_size);
  *elf_header = obj_base;
  if (!IsValidElf(*elf_header)) {
    fprintf(stderr, "Not a valid ELF file: %s\n", obj_file.c_str());
    return false;
  }
  return true;
}

// Get the endianness of ELF_HEADER. If it's invalid, return false.
template<typename ElfClass>
bool ElfEndianness(const typename ElfClass::Ehdr* elf_header,
                   bool* big_endian) {
  if (elf_header->e_ident[EI_DATA] == ELFDATA2LSB) {
    *big_endian = false;
    return true;
  }
  if (elf_header->e_ident[EI_DATA] == ELFDATA2MSB) {
    *big_endian = true;
    return true;
  }

  fprintf(stderr, "bad data encoding in ELF header: %d\n",
          elf_header->e_ident[EI_DATA]);
  return false;
}

// Given |left_abspath|, find the absolute path for |right_path| and see if the
// two absolute paths are the same.
bool IsSameFile(const char* left_abspath, const string& right_path) {
  char right_abspath[PATH_MAX];
  if (!realpath(right_path.c_str(), right_abspath))
    return false;
  return strcmp(left_abspath, right_abspath) == 0;
}

// Read the .gnu_debuglink and get the debug file name. If anything goes
// wrong, return an empty string.
string ReadDebugLink(const uint8_t* debuglink,
                     const size_t debuglink_size,
                     const bool big_endian,
                     const string& obj_file,
                     const std::vector<string>& debug_dirs) {
  // Include '\0' + CRC32 (4 bytes).
  size_t debuglink_len = strlen(reinterpret_cast<const char*>(debuglink)) + 5;
  debuglink_len = 4 * ((debuglink_len + 3) / 4);  // Round up to 4 bytes.

  // Sanity check.
  if (debuglink_len != debuglink_size) {
    fprintf(stderr, "Mismatched .gnu_debuglink string / section size: "
            "%zx %zx\n", debuglink_len, debuglink_size);
    return string();
  }

  char obj_file_abspath[PATH_MAX];
  if (!realpath(obj_file.c_str(), obj_file_abspath)) {
    fprintf(stderr, "Cannot resolve absolute path for %s\n", obj_file.c_str());
    return string();
  }

  std::vector<string> searched_paths;
  string debuglink_path;
  std::vector<string>::const_iterator it;
  for (it = debug_dirs.begin(); it < debug_dirs.end(); ++it) {
    const string& debug_dir = *it;
    debuglink_path = debug_dir + "/" +
                     reinterpret_cast<const char*>(debuglink);

    // There is the annoying case of /path/to/foo.so having foo.so as the
    // debug link file name. Thus this may end up opening /path/to/foo.so again,
    // and there is a small chance of the two files having the same CRC.
    if (IsSameFile(obj_file_abspath, debuglink_path))
      continue;

    searched_paths.push_back(debug_dir);
    int debuglink_fd = open(debuglink_path.c_str(), O_RDONLY);
    if (debuglink_fd < 0)
      continue;

    FDWrapper debuglink_fd_wrapper(debuglink_fd);

    // The CRC is the last 4 bytes in |debuglink|.
    const google_breakpad::Endianness endianness = big_endian ?
        google_breakpad::ENDIANNESS_BIG : google_breakpad::ENDIANNESS_LITTLE;
    google_breakpad::ByteReader byte_reader(endianness);
    uint32_t expected_crc =
        byte_reader.ReadFourBytes(&debuglink[debuglink_size - 4]);

    uint32_t actual_crc = 0;
    while (true) {
      const size_t kReadSize = 4096;
      char buf[kReadSize];
      ssize_t bytes_read = HANDLE_EINTR(read(debuglink_fd, &buf, kReadSize));
      if (bytes_read < 0) {
        fprintf(stderr, "Error reading debug ELF file %s.\n",
                debuglink_path.c_str());
        return string();
      }
      if (bytes_read == 0)
        break;
      actual_crc = google_breakpad::UpdateCrc32(actual_crc, buf, bytes_read);
    }
    if (actual_crc != expected_crc) {
      fprintf(stderr, "Error reading debug ELF file - CRC32 mismatch: %s\n",
              debuglink_path.c_str());
      continue;
    }

    // Found debug file.
    return debuglink_path;
  }

  // Not found case.
  fprintf(stderr, "Failed to find debug ELF file for '%s' after trying:\n",
          obj_file.c_str());
  for (it = searched_paths.begin(); it < searched_paths.end(); ++it) {
    const string& debug_dir = *it;
    fprintf(stderr, "  %s/%s\n", debug_dir.c_str(), debuglink);
  }
  return string();
}

//
// LoadSymbolsInfo
//
// Holds the state between the two calls to LoadSymbols() in case it's necessary
// to follow the .gnu_debuglink section and load debug information from a
// different file.
//
template<typename ElfClass>
class LoadSymbolsInfo {
 public:
  typedef typename ElfClass::Addr Addr;

  explicit LoadSymbolsInfo(const std::vector<string>& dbg_dirs) :
    debug_dirs_(dbg_dirs),
    has_loading_addr_(false) {}

  // Keeps track of which sections have been loaded so sections don't
  // accidentally get loaded twice from two different files.
  void LoadedSection(const string& section) {
    if (loaded_sections_.count(section) == 0) {
      loaded_sections_.insert(section);
    } else {
      fprintf(stderr, "Section %s has already been loaded.\n",
              section.c_str());
    }
  }

  // The ELF file and linked debug file are expected to have the same preferred
  // loading address.
  void set_loading_addr(Addr addr, const string& filename) {
    if (!has_loading_addr_) {
      loading_addr_ = addr;
      loaded_file_ = filename;
      return;
    }

    if (addr != loading_addr_) {
      fprintf(stderr,
              "ELF file '%s' and debug ELF file '%s' "
              "have different load addresses.\n",
              loaded_file_.c_str(), filename.c_str());
      assert(false);
    }
  }

  // Setters and getters
  const std::vector<string>& debug_dirs() const {
    return debug_dirs_;
  }

  string debuglink_file() const {
    return debuglink_file_;
  }
  void set_debuglink_file(string file) {
    debuglink_file_ = file;
  }

 private:
  const std::vector<string>& debug_dirs_; // Directories in which to
                                          // search for the debug ELF file.

  string debuglink_file_;  // Full path to the debug ELF file.

  bool has_loading_addr_;  // Indicate if LOADING_ADDR_ is valid.

  Addr loading_addr_;  // Saves the preferred loading address from the
                       // first call to LoadSymbols().

  string loaded_file_;  // Name of the file loaded from the first call to
                        // LoadSymbols().

  std::set<string> loaded_sections_;  // Tracks the Loaded ELF sections
                                      // between calls to LoadSymbols().
};

template<typename ElfClass>
bool LoadSymbols(const string& obj_file,
                 const bool big_endian,
                 const typename ElfClass::Ehdr* elf_header,
                 const bool read_gnu_debug_link,
                 LoadSymbolsInfo<ElfClass>* info,
                 const DumpOptions& options,
                 Module* module) {
  typedef typename ElfClass::Addr Addr;
  typedef typename ElfClass::Phdr Phdr;
  typedef typename ElfClass::Shdr Shdr;

  Addr loading_addr = GetLoadingAddress<ElfClass>(
      GetOffset<ElfClass, Phdr>(elf_header, elf_header->e_phoff),
      elf_header->e_phnum);
  module->SetLoadAddress(loading_addr);
  info->set_loading_addr(loading_addr, obj_file);

  // Allow filtering of extraneous debug information in partitioned libraries.
  // Such libraries contain debug information for all libraries extracted from
  // the same combined library, implying extensive duplication.
  vector<Module::Range> address_ranges = GetPtLoadSegmentRanges<ElfClass>(
      GetOffset<ElfClass, Phdr>(elf_header, elf_header->e_phoff),
      elf_header->e_phnum);
  module->SetAddressRanges(address_ranges);

  const Shdr* sections =
      GetOffset<ElfClass, Shdr>(elf_header, elf_header->e_shoff);
  const Shdr* section_names = sections + elf_header->e_shstrndx;
  const char* names =
      GetOffset<ElfClass, char>(elf_header, section_names->sh_offset);
  const char* names_end = names + section_names->sh_size;
  bool found_debug_info_section = false;
  bool found_usable_info = false;
  bool usable_info_parsed = false;

  if ((options.symbol_data & SYMBOLS_AND_FILES) ||
      (options.symbol_data & INLINES)) {
#ifndef NO_STABS_SUPPORT
    // Look for STABS debugging information, and load it if present.
    const Shdr* stab_section =
      FindElfSectionByName<ElfClass>(".stab", SHT_PROGBITS,
                                     sections, names, names_end,
                                     elf_header->e_shnum);
    if (stab_section) {
      const Shdr* stabstr_section = stab_section->sh_link + sections;
      if (stabstr_section) {
        found_debug_info_section = true;
        found_usable_info = true;
        info->LoadedSection(".stab");
        bool result = LoadStabs<ElfClass>(elf_header, stab_section, stabstr_section,
                                 big_endian, module);
        usable_info_parsed = usable_info_parsed || result;
        if (!result) {
          fprintf(stderr, "%s: \".stab\" section found, but failed to load"
                  " STABS debugging information\n", obj_file.c_str());
        }
      }
    }
#endif  // NO_STABS_SUPPORT

    // See if there are export symbols available.
    const Shdr* symtab_section =
        FindElfSectionByName<ElfClass>(".symtab", SHT_SYMTAB,
                                       sections, names, names_end,
                                       elf_header->e_shnum);
    const Shdr* strtab_section =
        FindElfSectionByName<ElfClass>(".strtab", SHT_STRTAB,
                                       sections, names, names_end,
                                       elf_header->e_shnum);
    if (symtab_section && strtab_section) {
      info->LoadedSection(".symtab");

      const uint8_t* symtab =
          GetOffset<ElfClass, uint8_t>(elf_header,
                                       symtab_section->sh_offset);
      const uint8_t* strtab =
          GetOffset<ElfClass, uint8_t>(elf_header,
                                       strtab_section->sh_offset);
      bool result =
          ELFSymbolsToModule(symtab,
                             symtab_section->sh_size,
                             strtab,
                             strtab_section->sh_size,
                             big_endian,
                             ElfClass::kAddrSize,
                             module);
      found_usable_info = found_usable_info || result;
    } else {
      // Look in dynsym only if full symbol table was not available.
      const Shdr* dynsym_section =
          FindElfSectionByName<ElfClass>(".dynsym", SHT_DYNSYM,
                                         sections, names, names_end,
                                         elf_header->e_shnum);
      const Shdr* dynstr_section =
          FindElfSectionByName<ElfClass>(".dynstr", SHT_STRTAB,
                                         sections, names, names_end,
                                         elf_header->e_shnum);
      if (dynsym_section && dynstr_section) {
        info->LoadedSection(".dynsym");

        const uint8_t* dynsyms =
            GetOffset<ElfClass, uint8_t>(elf_header,
                                         dynsym_section->sh_offset);
        const uint8_t* dynstrs =
            GetOffset<ElfClass, uint8_t>(elf_header,
                                         dynstr_section->sh_offset);
        bool result =
            ELFSymbolsToModule(dynsyms,
                               dynsym_section->sh_size,
                               dynstrs,
                               dynstr_section->sh_size,
                               big_endian,
                               ElfClass::kAddrSize,
                               module);
        found_usable_info = found_usable_info || result;
      }
    }

    // Only Load .debug_info after loading symbol table to avoid duplicate
    // PUBLIC records.
    // Look for DWARF debugging information, and load it if present.
    const Shdr* dwarf_section =
      FindElfSectionByName<ElfClass>(".debug_info", SHT_PROGBITS,
                                     sections, names, names_end,
                                     elf_header->e_shnum);

    // .debug_info section type is SHT_PROGBITS for mips on pnacl toolchains,
    // but MIPS_DWARF for regular gnu toolchains, so both need to be checked
    if (elf_header->e_machine == EM_MIPS && !dwarf_section) {
      dwarf_section =
        FindElfSectionByName<ElfClass>(".debug_info", SHT_MIPS_DWARF,
                                       sections, names, names_end,
                                       elf_header->e_shnum);
    }

    if (dwarf_section) {
      found_debug_info_section = true;
      found_usable_info = true;
      info->LoadedSection(".debug_info");
      bool result = LoadDwarf<ElfClass>(obj_file, elf_header, big_endian,
                               options.handle_inter_cu_refs,
                               options.symbol_data & INLINES, module);
      usable_info_parsed = usable_info_parsed || result;
      if (!result){
        fprintf(stderr, "%s: \".debug_info\" section found, but failed to load "
                "DWARF debugging information\n", obj_file.c_str());
      }
    }
  }

  if (options.symbol_data & CFI) {
    // Dwarf Call Frame Information (CFI) is actually independent from
    // the other DWARF debugging information, and can be used alone.
    const Shdr* dwarf_cfi_section =
        FindElfSectionByName<ElfClass>(".debug_frame", SHT_PROGBITS,
                                       sections, names, names_end,
                                       elf_header->e_shnum);

    // .debug_frame section type is SHT_PROGBITS for mips on pnacl toolchains,
    // but MIPS_DWARF for regular gnu toolchains, so both need to be checked
    if (elf_header->e_machine == EM_MIPS && !dwarf_cfi_section) {
      dwarf_cfi_section =
          FindElfSectionByName<ElfClass>(".debug_frame", SHT_MIPS_DWARF,
                                        sections, names, names_end,
                                        elf_header->e_shnum);
    }

    if (dwarf_cfi_section) {
      // Ignore the return value of this function; even without call frame
      // information, the other debugging information could be perfectly
      // useful.
      info->LoadedSection(".debug_frame");
      bool result =
          LoadDwarfCFI<ElfClass>(obj_file, elf_header, ".debug_frame",
                                 dwarf_cfi_section, false, 0, 0, big_endian,
                                 module);
      found_usable_info = found_usable_info || result;
    }

    // Linux C++ exception handling information can also provide
    // unwinding data.
    const Shdr* eh_frame_section =
        FindElfSectionByName<ElfClass>(".eh_frame", SHT_PROGBITS,
                                       sections, names, names_end,
                                       elf_header->e_shnum);
    if (eh_frame_section) {
      // Pointers in .eh_frame data may be relative to the base addresses of
      // certain sections. Provide those sections if present.
      const Shdr* got_section =
          FindElfSectionByName<ElfClass>(".got", SHT_PROGBITS,
                                         sections, names, names_end,
                                         elf_header->e_shnum);
      const Shdr* text_section =
          FindElfSectionByName<ElfClass>(".text", SHT_PROGBITS,
                                         sections, names, names_end,
                                         elf_header->e_shnum);
      info->LoadedSection(".eh_frame");
      // As above, ignore the return value of this function.
      bool result =
          LoadDwarfCFI<ElfClass>(obj_file, elf_header, ".eh_frame",
                                 eh_frame_section, true,
                                 got_section, text_section, big_endian, module);
      found_usable_info = found_usable_info || result;
    }
  }

  if (!found_debug_info_section) {
    fprintf(stderr, "%s: file contains no debugging information"
            " (no \".stab\" or \".debug_info\" sections)\n",
            obj_file.c_str());

    // Failed, but maybe there's a .gnu_debuglink section?
    if (read_gnu_debug_link) {
      const Shdr* gnu_debuglink_section
          = FindElfSectionByName<ElfClass>(".gnu_debuglink", SHT_PROGBITS,
                                           sections, names,
                                           names_end, elf_header->e_shnum);
      if (gnu_debuglink_section) {
        if (!info->debug_dirs().empty()) {
          const uint8_t* debuglink_contents =
              GetOffset<ElfClass, uint8_t>(elf_header,
                                           gnu_debuglink_section->sh_offset);
          string debuglink_file =
              ReadDebugLink(debuglink_contents,
                            gnu_debuglink_section->sh_size,
                            big_endian,
                            obj_file,
                            info->debug_dirs());
          info->set_debuglink_file(debuglink_file);
        } else {
          fprintf(stderr, ".gnu_debuglink section found in '%s', "
                  "but no debug path specified.\n", obj_file.c_str());
        }
      } else {
        fprintf(stderr, "%s does not contain a .gnu_debuglink section.\n",
                obj_file.c_str());
      }
    } else {
      // Return true if some usable information was found, since the caller
      // doesn't want to use .gnu_debuglink.
      return found_usable_info;
    }

    // No debug info was found, let the user try again with .gnu_debuglink
    // if present.
    return false;
  }

  return usable_info_parsed;
}

// Return the breakpad symbol file identifier for the architecture of
// ELF_HEADER.
template<typename ElfClass>
const char* ElfArchitecture(const typename ElfClass::Ehdr* elf_header) {
  typedef typename ElfClass::Half Half;
  Half arch = elf_header->e_machine;
  switch (arch) {
    case EM_386:        return "x86";
    case EM_ARM:        return "arm";
    case EM_AARCH64:    return "arm64";
    case EM_MIPS:       return "mips";
    case EM_PPC64:      return "ppc64";
    case EM_PPC:        return "ppc";
    case EM_S390:       return "s390";
    case EM_SPARC:      return "sparc";
    case EM_SPARCV9:    return "sparcv9";
    case EM_X86_64:     return "x86_64";
    case EM_RISCV:      return "riscv";
    default: return NULL;
  }
}

template<typename ElfClass>
bool SanitizeDebugFile(const typename ElfClass::Ehdr* debug_elf_header,
                       const string& debuglink_file,
                       const string& obj_filename,
                       const char* obj_file_architecture,
                       const bool obj_file_is_big_endian) {
  const char* debug_architecture =
      ElfArchitecture<ElfClass>(debug_elf_header);
  if (!debug_architecture) {
    fprintf(stderr, "%s: unrecognized ELF machine architecture: %d\n",
            debuglink_file.c_str(), debug_elf_header->e_machine);
    return false;
  }
  if (strcmp(obj_file_architecture, debug_architecture)) {
    fprintf(stderr, "%s with ELF machine architecture %s does not match "
            "%s with ELF architecture %s\n",
            debuglink_file.c_str(), debug_architecture,
            obj_filename.c_str(), obj_file_architecture);
    return false;
  }
  bool debug_big_endian;
  if (!ElfEndianness<ElfClass>(debug_elf_header, &debug_big_endian))
    return false;
  if (debug_big_endian != obj_file_is_big_endian) {
    fprintf(stderr, "%s and %s does not match in endianness\n",
            obj_filename.c_str(), debuglink_file.c_str());
    return false;
  }
  return true;
}

template<typename ElfClass>
bool InitModuleForElfClass(const typename ElfClass::Ehdr* elf_header,
                           const string& obj_filename,
                           const string& obj_os,
                           const string& module_id,
                           scoped_ptr<Module>& module,
                           bool enable_multiple_field) {
  PageAllocator allocator;
  wasteful_vector<uint8_t> identifier(&allocator, kDefaultBuildIdSize);
  if (!FileID::ElfFileIdentifierFromMappedFile(elf_header, identifier)) {
    fprintf(stderr, "%s: unable to generate file identifier\n",
            obj_filename.c_str());
    return false;
  }

  const char* architecture = ElfArchitecture<ElfClass>(elf_header);
  if (!architecture) {
    fprintf(stderr, "%s: unrecognized ELF machine architecture: %d\n",
            obj_filename.c_str(), elf_header->e_machine);
    return false;
  }

  char name_buf[NAME_MAX] = {};
  std::string name = google_breakpad::ElfFileSoNameFromMappedFile(
                         elf_header, name_buf, sizeof(name_buf))
                         ? name_buf
                         : google_breakpad::BaseName(obj_filename);

  // Use the provided module_id
  string id = module_id.empty()
    // Add an extra "0" at the end.  PDB files on Windows have an 'age'
    // number appended to the end of the file identifier; this isn't
    // really used or necessary on other platforms, but be consistent.
    ? FileID::ConvertIdentifierToUUIDString(identifier) + "0"
    : module_id;

  // This is just the raw Build ID in hex.
  string code_id = FileID::ConvertIdentifierToString(identifier);

  module.reset(new Module(name, obj_os, architecture, id, code_id,
                          enable_multiple_field));

  return true;
}

template<typename ElfClass>
bool ReadSymbolDataElfClass(const typename ElfClass::Ehdr* elf_header,
                            const string& obj_filename,
                            const string& obj_os,
                            const string& module_id,
                            const std::vector<string>& debug_dirs,
                            const DumpOptions& options,
                            Module** out_module) {
  typedef typename ElfClass::Ehdr Ehdr;

  *out_module = NULL;

  scoped_ptr<Module> module;
  if (!InitModuleForElfClass<ElfClass>(elf_header, obj_filename, obj_os, module_id,
                                      module, options.enable_multiple_field)) {
    return false;
  }

  // Figure out what endianness this file is.
  bool big_endian;
  if (!ElfEndianness<ElfClass>(elf_header, &big_endian))
    return false;

  LoadSymbolsInfo<ElfClass> info(debug_dirs);
  if (!LoadSymbols<ElfClass>(obj_filename, big_endian, elf_header,
                             !debug_dirs.empty(), &info,
                             options, module.get())) {
    const string debuglink_file = info.debuglink_file();
    if (debuglink_file.empty())
      return false;

    // Load debuglink ELF file.
    fprintf(stderr, "Found debugging info in %s\n", debuglink_file.c_str());
    MmapWrapper debug_map_wrapper;
    Ehdr* debug_elf_header = NULL;
    if (!LoadELF(debuglink_file, &debug_map_wrapper,
                 reinterpret_cast<void**>(&debug_elf_header)) ||
        !SanitizeDebugFile<ElfClass>(debug_elf_header, debuglink_file,
                                     obj_filename,
                                     module->architecture().c_str(),
                                     big_endian)) {
      return false;
    }

    if (!LoadSymbols<ElfClass>(debuglink_file, big_endian,
                               debug_elf_header, false, &info,
                               options, module.get())) {
      return false;
    }
  }

  *out_module = module.release();
  return true;
}

}  // namespace

namespace google_breakpad {

// Not explicitly exported, but not static so it can be used in unit tests.
bool ReadSymbolDataInternal(const uint8_t* obj_file,
                            const string& obj_filename,
                            const string& obj_os,
                            const string& module_id,
                            const std::vector<string>& debug_dirs,
                            const DumpOptions& options,
                            Module** module) {
  if (!IsValidElf(obj_file)) {
    fprintf(stderr, "Not a valid ELF file: %s\n", obj_filename.c_str());
    return false;
  }

  int elfclass = ElfClass(obj_file);
  if (elfclass == ELFCLASS32) {
    return ReadSymbolDataElfClass<ElfClass32>(
        reinterpret_cast<const Elf32_Ehdr*>(obj_file), obj_filename, obj_os,
        module_id, debug_dirs, options, module);
  }
  if (elfclass == ELFCLASS64) {
    return ReadSymbolDataElfClass<ElfClass64>(
        reinterpret_cast<const Elf64_Ehdr*>(obj_file), obj_filename, obj_os,
        module_id, debug_dirs, options, module);
  }

  return false;
}

bool WriteSymbolFile(const string& load_path,
                     const string& obj_file,
                     const string& obj_os,
                     const string& module_id,
                     const std::vector<string>& debug_dirs,
                     const DumpOptions& options,
                     std::ostream& sym_stream) {
  Module* module;
  if (!ReadSymbolData(load_path, obj_file, obj_os, module_id, debug_dirs, options,
                      &module))
    return false;

  bool result = module->Write(sym_stream, options.symbol_data, options.preserve_load_address);
  delete module;
  return result;
}

// Read the selected object file's debugging information, and write out the
// header only to |stream|. Return true on success; if an error occurs, report
// it and return false.
bool WriteSymbolFileHeader(const string& load_path,
                           const string& obj_file,
                           const string& obj_os,
                           const string& module_id,
                           std::ostream& sym_stream) {
  MmapWrapper map_wrapper;
  void* elf_header = NULL;
  if (!LoadELF(load_path, &map_wrapper, &elf_header)) {
    fprintf(stderr, "Could not load ELF file: %s\n", obj_file.c_str());
    return false;
  }

  if (!IsValidElf(elf_header)) {
    fprintf(stderr, "Not a valid ELF file: %s\n", obj_file.c_str());
    return false;
  }

  int elfclass = ElfClass(elf_header);
  scoped_ptr<Module> module;
  if (elfclass == ELFCLASS32) {
    if (!InitModuleForElfClass<ElfClass32>(
        reinterpret_cast<const Elf32_Ehdr*>(elf_header), obj_file, obj_os,
        module_id, module, /*enable_multiple_field=*/false)) {
      fprintf(stderr, "Failed to load ELF module: %s\n", obj_file.c_str());
      return false;
    }
  } else if (elfclass == ELFCLASS64) {
    if (!InitModuleForElfClass<ElfClass64>(
        reinterpret_cast<const Elf64_Ehdr*>(elf_header), obj_file, obj_os,
        module_id, module, /*enable_multiple_field=*/false)) {
      fprintf(stderr, "Failed to load ELF module: %s\n", obj_file.c_str());
      return false;
    }
  } else {
    fprintf(stderr, "Unsupported module file: %s\n", obj_file.c_str());
    return false;
  }

  return module->Write(sym_stream, ALL_SYMBOL_DATA);
}

bool ReadSymbolData(const string& load_path,
                    const string& obj_file,
                    const string& obj_os,
                    const string& module_id,
                    const std::vector<string>& debug_dirs,
                    const DumpOptions& options,
                    Module** module) {
  MmapWrapper map_wrapper;
  void* elf_header = NULL;
  if (!LoadELF(load_path, &map_wrapper, &elf_header))
    return false;

  return ReadSymbolDataInternal(reinterpret_cast<uint8_t*>(elf_header),
                                obj_file, obj_os, module_id, debug_dirs, options, module);
}

}  // namespace google_breakpad
